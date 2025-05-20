import os, json, re
import spacy
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse  # render에 10분 단위 Ping 보내기를 위해 추가
from pydantic import BaseModel
# 아래 api_key= 까지는 .env 파일에서 OpenAI키를 불러오기 관련 부분 
from openai import OpenAI
from dotenv import load_dotenv

# ◎ 환경 설정
load_dotenv()

api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY is not set in environment variables.")
client = OpenAI(api_key=api_key)

app = FastAPI()  # FastAPI() 객체를 생성해서 이후 라우팅에 사용
nlp = spacy.load("en_core_web_sm")  # spaCy 관련 설정, (englihs_core모델_web기반_small)

# ◎ 메모리 구조
memory = {
#    "characters": [],
    "symbols_by_level": {},
}

# ◎ 심볼 매핑
role_to_symbol = {
    "verb": "○",
    "object": "□",
    "direct object": "□",
    "indirect object": "□",
    "prepositional object": "□",
    "preposition": "▽",
    "conjunction": "◇"
}

# ◎ 요청/응답 목록
class AnalyzeRequest(BaseModel):   # 사용자가 보낼 요청(sentence) 정의
    sentence: str

class AnalyzeResponse(BaseModel):  # 응답으로 돌려줄 데이터(sentence, diagramming) 정의
    sentence: str
    diagramming: str               # "     ○______□__[         "

class ParseRequest(BaseModel):     # spaCy 관련 설정
    text: str

# ◎ 문자 저장
def init_memorys (sentence: str):
#    memory["characters"] = list(sentence)        # characters에 sentence의 글자 한글자씩 채우기
    memory["symbols_by_level"] = {}  # 문장마다 새로 초기화
    memory["sentence_length"] = len(sentence)  # 도식 길이 추적용 (줄 길이 통일)


# ◎ GPT 파스함수
def gpt_parse(sentence: str):
    prompt = f"""
Analyze the following English sentence and return a JSON array.

Each item must include these 10 fields, in this exact order:
1. "idx" – character index (spaCy token.idx)
2. "text" – the word itself
3. "role" – one of the fixed roles (see below)
4. "combine" – optional; only for main verbs and prepositions
5. "level" – depth in dependency tree

---

🔹 Allowed "role" values and their conditions:
- subject: dep_ is "nsubj" or "nsubjpass"
- verb: dep_ is "ROOT" and pos is "VERB", excluding auxiliary verbs (only the main verb per clause)
- object: dep_ is "dobj" or "obj" - SVO
- subject complement: dep_ is "attr" or "acomp" - SVC
- object complement: dep_ is "xcomp", "oprd", or "ccomp" - SVOC
- indirect object: dep_ is "iobj" - SVOO
- direct object: dep_ is "dobj" and "iobj" also exists - SVOO
- preposition: dep_ is "prep"
- prepositional object: dep_ is "pobj"
- conjunction: dep_ is "cc" or "mark"

❌ Do not invent new roles.  
❌ Do not use labels like "subject noun complement" or "relative pronoun".  
❌ If a token doesn’t match any of the above, omit the "role" field.

---

🔹 When assigning "combine", only include tokens that meet one of the following structural relationships:

- verb → subject complement       (SVC)
- verb → object                   (SVO)
- verb → indirect object          (SVOO)
- indirect object → direct object (SVOO)
- object → object complement      (SVOC)
- preposition → prepositional object

Each "combine" must reflect an underline connection in DrawEnglish diagrams.  
Do NOT include modifiers, adverbs, prepositions, or conjunctions in "combine".
If none of the above applies, omit the "combine" field entirely.

🔹 Level Rules

Assign a "level" value to each token to indicate its structural depth in the sentence.

- level 0: main clause
- level 1+: subordinate clauses, infinitive phrases, or verbals

A new level begins when any of the following structural triggers appears:
- Subordinating conjunctions (e.g., that, because, if)
- Infinitives (to + verb)
- Gerunds (verb-ing)
- Present participles
- Past participles
- Relative pronouns (who, which, that)
- Wh-words (what, why, where, how)
- Compound wh-words (whoever, whatever, however, etc.)

When a new level begins:
- The **first word** (trigger) must be assigned level `n - 0.5`
- All other words in that phrase/clause receive level `n`

This level system is used to separate clauses and reduce visual confusion in sentence diagrams.
Combine links must only occur within the same level.

> ✅ **Optimization Rule:**  
> If the sentence contains **no structural triggers**, assign `"level": 0` to **all items**.  
> You do **not** need to check for deeper structures in that case.


🔹 Example format:
[
  {{
    "idx": 5, "text": "elected", "role": "verb", 
    "combine": [ {{ "text": "him", "role": "object" }}, {{ "text": "president", "role": "object complement" }} ],
    "level": 0
  }}
]

---

Sentence: "{sentence}"

Return ONLY the raw JSON array.  
Do not explain anything. Do not include any text outside the array.
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
#        print("[GPT RESPONSE]", content)  # GPT 응답 직접 확인
        return json.loads(content)
    except Exception as e:
        print("[ERROR] GPT parsing failed:", e)
        print("[RAW CONTENT]", content)  # 문제가 된 원본 그대로 출력
        return []


# ◎ symbols 메모리에 심볼들 저장하기
def apply_symbols(parsed):
    symbols_by_level = memory["symbols_by_level"]
    line_length = memory["sentence_length"]

    for item in parsed:
        idx = item.get("idx", -1)
        role = item.get("role", "").lower()
        pos = item.get("pos", "").upper()
        level = item.get("level")

        if idx < 0 or level is None:
            continue

        # ✅ 0.5처럼 경계 레벨은 두 줄에 심볼 찍기
        levels = [level]
        if isinstance(level, float) and level % 1 == 0.5:
            levels = [int(level), int(level) + 1]

        # ✅ 보어 심볼 결정
        if role in ["subject complement", "object complement"]:
            if pos in ["NOUN", "PROPN", "PRON"]:
                symbol = "["
            elif pos == "ADJ":
                symbol = "("
            else:
                symbol = None
        else:
            symbol = role_to_symbol.get(role)

        for lvl in levels:
            line = symbols_by_level.setdefault(lvl, [" " for _ in range(line_length)])
            if 0 <= idx < len(line) and line[idx] == " " and symbol:
                line[idx] = symbol



# ◎ memory["symbols"] 내용을 출력하기 위해 만든 함수
def symbols_to_diagram():
    output_lines = []
    for level in sorted(memory["symbols_by_level"]):
        line = ''.join(memory["symbols_by_level"][level])
        output_lines.append(line)
    return '\n'.join(output_lines)


# ◎ 디버깅용 테스트 함수
def test(sentence: str, use_gpt: bool = True):
    print(f"\n📘 Sentence: {sentence}")
    if use_gpt:
        parsed = gpt_parse(sentence)
    else:
        parsed = []  # or mock GPT result for offline test
    if not parsed:
        print("❌ No GPT parsing result.")
    else:
        print("\n📊 GPT Parsing Result:")
    for item in parsed:
        combine = item.get("combine")
        if combine:
            combine_str = "[" + ', '.join(f"{c.get('text')}:{c.get('role')}" for c in combine) + "]"
        else:
            combine_str = "None"
        print(
            f"● idx({item.get('idx')}), text({item.get('text')}), role({item.get('role')}), "
            f"combine({combine_str}), level({item.get('level')})"
        )

    print("\n📘 spaCy Parsing Result:")
    doc = nlp(sentence)
    for token in doc:
        morph = token.morph.to_dict()
        print(
            f"● idx({token.idx}), text({token.text}), pos({token.pos_}), tag({token.tag_}), dep({token.dep_}), "
            f"head({token.head.text}), tense({morph.get('Tense')}), form({morph.get('VerbForm')}), "
            f"voice({morph.get('Voice')}), morph({morph})"
        )

    init_memorys(sentence)
    apply_symbols(parsed)
    print("\n🛠 Sentence Diagram:")
    index_line = ''.join([str(i % 10) for i in range(len(sentence))])
    print(index_line)
    print(sentence)
    print(symbols_to_diagram())


# 초간단 임시 테스트1 함수
def test1():
    print(type(symbols_to_diagram()))



# ◎ 모듈 외부 사용을 위한 export
__all__ = [
    "init_memorys",
    "gpt_parse"
    "apply_symbols",
    "test"
]

# ◎ 분석 API 엔드포인트
@app.post("/analyze", response_model=AnalyzeResponse)  # sentence를 받아 "sentence"와 "diagramming" 리턴
async def analyze(request: AnalyzeRequest):            # sentence를 받아 다음 처리로 넘김
    init_memorys(request.sentence)                     # 이 함수로 메모리 내용 채움 또는 초기화
    parsed = gpt_parse(request.sentence)               # GPT의 파싱결과를 parsed에 저장
    apply_symbols(parsed)                              # parsed 결과에 따라 심볼들을 메모리에 저장장
    return {"sentence": request.sentence, "diagramming": symbols_to_diagram()}

# ◎ spaCy 파싱 관련
@app.post("/parse")
def parse_text(req: ParseRequest):
    doc = nlp(req.text)
    result = []
    for token in doc:
        morph = token.morph.to_dict()
        result.append({
            "idx": token.idx, "text": token.text, "pos": token.pos_, "tag": token.tag_, "dep": token.dep_,
            "head": token.head.text, "tense": morph.get("Tense"), "form": morph.get("VerbForm"),
            "voice": morph.get("Voice"), "morph": morph
        })
    return {"result": result}

# ◎ 커스텀 OpenAPI JSON 제공 엔드포인트
# FastAPI에서 custom-openapi.json 엔드포인트를 만들어서 GPTs에서 사용할 수 있도록 함.
@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openapi.json"))
    # file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")

# ◎ 아래 엔드포인트는 GET /ping 요청에 대해 {"message": "pong"} 응답을 준다.
@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "pong"}, status_code=200)

