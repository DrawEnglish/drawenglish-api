import os, json, re
from fastapi import FastAPI  # FastAPI는 Python 기반의 웹 프레임워크로, Re=EST API를 만들때 매우 간단하고 빠름
from fastapi.responses import JSONResponse  # render에 10분 단위 Ping 보내기를 위해 추가
from pydantic import BaseModel  # BaseModeld=은 FastAPI에서 request/response의 데이터 검증과 자동 문서화를 위해 사용되는 클래스(Pydantic제공)
# 아래 api_key= 까지는 .env 파일에서 OpenAI키를 불러와 설정하는 부분 
from openai import OpenAI
from dotenv import load_dotenv

# 0. 환경 설정
load_dotenv()

api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY is not set in environment variables.")
client = OpenAI(api_key=api_key)

app = FastAPI()  # FastAPI() 객체를 생성해서 이후 라우팅에 사용

# 1. 메모리 구조
memory = {
    "characters": [],
    "symbols": [],
    "char_lower": "",
    "word_positions": []
}

# 2. 심볼 매핑
role_to_symbol = {
    "verb": "○",
    "object": "□",
    "subject noun complement": "[",
    "object noun complement": "[",
    "subject adjective complement": "(",
    "object adjective complement": "(",
    "preposition": "▽",
    "conjunction": "◇"
}

# 3. 요청/응답 목록
class AnalyzeRequest(BaseModel):  # 사용자가 보낼 요청(sentence) 정의
    sentence: str

class AnalyzeResponse(BaseModel):  # 응답으로 돌려줄 데이터(sentence, diagramming) 정의
    sentence: str
    diagramming: str

# 4. 문자 저장
def store_characters(sentence: str):
    memory["characters"] = list(sentence)
    memory["symbols"] = [" " for _ in sentence]
    memory["char_lower"] = sentence.lower()
    memory["word_positions"] = [
        {"token": m.group(), "index": m.start(), "used": False}
        for m in re.finditer(r'\b\w+\b', memory["char_lower"])
    ]

# 5. GPT 파시함수
def gpt_parse(sentence: str, verbose: bool = True):
    prompt = f"""
Analyze the following English sentence and return a JSON array.

Each item must have:
- \"word\": the word itself
- \"role\": one of:
  [subject, verb, object, subject noun complement, object noun complement, subject adjective complement, object adjective complement, preposition, conjunction]
- (optional) \"combine\": only for main verbs and prepositions. An array of objects with:
  {{ "word": "..." , "role": "..." }}

---

🔹 RULES:

1. ✅ Use only **one main verb per clause**.
   - If auxiliary verbs exist (e.g., "will have been eating"), only the final **main verb** ("eating") is labeled `"role": "verb"`.

2. ✅ Use `"combine"` only in these structures:
   - SVO → 1 object
   - SVOO → 2 objects
   - SVOC → object + complement
   - SVC → 1 subject complement
   - SV → ❌ do not include `"combine"` at all

3. ✅ For `"combine"`, only include objects or complements **directly governed by the verb or preposition**.
   - Never include prepositions, modifiers, or adverbs in `"combine"`.
   - Do not include any prepositional phrase or adverbial.

4. ✅ If the object/complement is a **phrase or clause** (e.g., to-infinitive, gerund, participial phrase, that-clause):
   → include only the **first word** of that phrase in the `"combine"` list.

5. ✅ **Prepositions** and their **objects** must be labeled separately.
   - If the preposition governs a noun or noun phrase, it should use `"combine"` to include only the first meaningful noun.
   - Do not include prepositions or their objects in the main verb's `"combine"`.

6. ❌ **Never label these function words** as `"preposition"` or `"object"`:
   - **Articles**: a, an, the
   - **Possessives**: my, your, his, her, its, our, their
   - **Modifiers/Adverbs**: very, really, too, also, quickly, fast, slowly, etc.
   → These should be ignored unless they act as main subject/object/complement.

   🔍 Especially:
   - Words like `"fast"`, `"quickly"` must NEVER be labeled as `"object"`.
   - If unsure, label them as `"adverb"` or omit from the JSON.

7. ✅ Conjunctions (e.g., "and", "that", "because") should be labeled `"conjunction"`.

8. ✅ For noun or adjective phrases like "the big red ball" or "my friend", always provide only the **head word** in `"combine"` (e.g., "ball", "friend", not "the ball" or "my friend").

   🔍 Examples:
   - Instead of: {{ "word": "the race", "role": "object" }}
     Use:        {{ "word": "race", "role": "object" }}

   - Instead of: {{ "word": "my friend", "role": "subject noun complement" }}
     Use:        {{ "word": "friend", "role": "subject noun complement" }}

9. ✅ In relative clauses (e.g., "The boy who won the race..."), treat the clause as a full SVO structure.
   - Label the verb (e.g., "won") as "verb"
   - Label the object (e.g., "race") as "object"
   - If the verb governs an object/complement, include a proper "combine" list as usual

10. ✅ Relative pronouns (e.g., "who", "which", "that" when referring to nouns) should be labeled "relative pronoun".
   - Do NOT label them as "conjunction".

---

Sentence: "{sentence}"

Return ONLY the raw JSON array. Do not explain anything. Do not include any text outside the array.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert sentence analyzer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        content = response.choices[0].message.content
        print("[GPT RESPONSE]", content)  # GPT 응답 직접 확인
        return json.loads(content)
    except Exception as e:
        print("[ERROR] GPT parsing failed:", e)
        print("[RAW CONTENT]", content)  # 문제가 된 원본 그대로 출력
        return []

# 6. 심볼 및 발행선 적용
def apply_symbols(parsed):
    line = memory["char_lower"]
    symbols = memory["symbols"]
    word_positions = memory["word_positions"]

    def find_unused(word):
        for pos in word_positions:
            if pos["token"] == word.lower() and not pos["used"]:
                pos["used"] = True
                return pos["index"]
        return -1

    for item in parsed:
        word = item["word"].lower()
        role = item["role"].lower()
        symbol = role_to_symbol.get(role)
        idx = find_unused(word)
        if symbol and idx != -1 and symbols[idx] == " ":
            symbols[idx] = symbol

        # ✅ combine 처리 (단, 전치사나 접조사는 연결선 그리지 않음)
        if role == "verb" and "combine" in item:
            for target in item["combine"]:
                t_word = target["word"].lower()
                t_role = target["role"].lower()

                # ⛔ 전치사, 접조사는 연결선 제외
                if t_role in ["preposition", "conjunction"]:
                    continue

                t_symbol = role_to_symbol.get(t_role)
                t_idx = find_unused(t_word)
                if t_idx != -1 and t_symbol and symbols[t_idx] == " ":
                    symbols[t_idx] = t_symbol
                    _connect(symbols, idx, t_idx)

        # ✅ 전치사에 combine이 있다면 (예: on → table)
        if role == "preposition" and "combine" in item:
            for target in item["combine"]:
                t_word = target["word"].lower()
                t_role = target["role"].lower()
                t_symbol = role_to_symbol.get(t_role)
                t_idx = find_unused(t_word)
                if t_idx != -1 and t_symbol and symbols[t_idx] == " ":
                    symbols[t_idx] = t_symbol
                    _connect(symbols, idx, t_idx)

    # ✅ 일반적인 전치사 + 목적에 관한 구조 처리 (combine 없이 나오는 경우 대비)
    for i in range(len(parsed) - 1):
        cur, nxt = parsed[i], parsed[i + 1]
        if cur["role"] == "preposition" and nxt["role"] == "object":
            c_idx = find_unused(cur["word"])
            n_idx = find_unused(nxt["word"])
            if c_idx != -1 and symbols[c_idx] == " ":
                symbols[c_idx] = role_to_symbol["preposition"]
            if n_idx != -1 and symbols[n_idx] == " ":
                symbols[n_idx] = role_to_symbol["object"]
            if c_idx != -1 and n_idx != -1:
                _connect(symbols, c_idx, n_idx)

# 7. 연결 함수
def _connect(symbols, start, end):
    if start > end:
        start, end = end, start
    for i in range(start + 1, end):
        if symbols[i] == " ":
            symbols[i] = "_"

# 8. Í# 8. \xcd9c력 함수
def print_diagram():
    diagram = diagram_filter_clean(memory["symbols"])  
    return f"\n{''.join(memory['characters'])}\n{''.join(diagram)}\n"  
    #◇,▽후 1칸 보정 필요 없을시 위 2줄은 아래 1줄로 치환
    #return f"\n{''.join(memory['characters'])}\n{''.join(memory['symbols'])}\n"


# 8-1. 출력 복지 함수 (◇, ▽ 뒤 1칸 무시) #◇,▽후 1칸 보정 필요 없을시 이 함수 전체 필요 없음음
def diagram_filter_clean(diagram):
    cleaned = []
    skip_next = False
    for ch in diagram:
        if skip_next:
            skip_next = False
            continue
        cleaned.append(ch)
        if ch in {'▽', '◇'}:
            skip_next = True
    return cleaned

# 9. 디버깅용 테스트 함수
def test(sentence: str, verbose: bool = True):
    if not verbose:
        import builtins
        builtins.print = lambda *args, **kwargs: None

    print("\n==============================")
    print("🔹 입력 문장:", sentence)

    store_characters(sentence)
    parsed = gpt_parse(sentence, verbose=verbose)

    print("\n📊 Parsed JSON:")
    print(json.dumps(parsed, indent=2))

    apply_symbols(parsed)

    print("\n🔨 Diagram:")
    print(print_diagram())

    print("\n🔢 인덱스:")
    index_line = ''.join([str(i % 10) for i in range(len(memory['characters']))])
    print(index_line)
    print(''.join(memory['characters']))
    print(''.join(memory['symbols']))

    print("\n🔍 단어 위치:")
    for pos in memory["word_positions"]:
        print(f"- {pos['token']:10s} at index {pos['index']}")

    print("==============================\n")

# 10. 모듈 외부 사용을 위한 export
__all__ = [
    "store_characters",
    "gpt_parse",
    "apply_symbols",
    "print_diagram",
    "test"
]

# 11. 분석 API 엔드포인트
@app.post("/analyze", response_model=AnalyzeResponse)  # 문장을 받아서 그에 대한 "문장 구조도(다이어그램)"를 응답으로 리턴
async def analyze(request: AnalyzeRequest):
    store_characters(request.sentence)
    parsed = gpt_parse(request.sentence)
    apply_symbols(parsed)
    return {"sentence": request.sentence, "diagramming": print_diagram()}

# 12. 커스텀 OpenAPI JSON 제공 엔드포인트
# FastAPI에서 custom-openapi.json 엔드포인트를 만들어서 GPTs에서 사용할 수 있도록 함.
@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openapi.json"))
    # file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")

# 아래 엔드포인트는 GET /ping 요청에 대해 {"message": "pong"} 응답을 준다.
@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "pong"}, status_code=200)

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

