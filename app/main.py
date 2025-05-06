from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os, json, re
from dotenv import load_dotenv

# 0. 환경 설정
load_dotenv()
client = OpenAI()
app = FastAPI()

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

# 3. 요청/응답 모델
class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
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

# 5. GPT 파싱 함수
def gpt_parse(sentence: str, verbose: bool = True):
    prompt = f"""
Analyze the following sentence.

Return a JSON array with one item per word.
Each item must have:
- word: the word itself
- role: one of [subject, verb, object, subject noun complement, object noun complement, subject adjective complement, object adjective complement, preposition, conjunction]
- (optional) combine: for verbs only. A list of objects or complements the verb connects to, as JSON objects with 'word' and 'role'.

Rules:
1. Only ONE verb per clause (the main verb, last word in a verb group).
2. For SVC/SVO/SVOO/SVOC, provide combine as:
   - SVC: 1 complement
   - SVO: 1 object
   - SVOO: 2 objects
   - SVOC: object + complement
3. If a complement or object is a clause or phrase (e.g., to-infinitive, gerund, participle, that-clause), only list its starting word.
4. For noun/adjective phrases (e.g., "the big red ball"), only list the head word (e.g., "ball").
5. Preposition + object must be marked separately.
6. Conjunctions are labeled but not connected.

Sentence: \"{sentence}\"

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

# 6. 심볼 및 밑줄 적용
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

        if role == "verb" and "combine" in item:
            for target in item["combine"]:
                t_word = target["word"].lower()
                t_role = target["role"].lower()
                t_symbol = role_to_symbol.get(t_role)
                t_idx = find_unused(t_word)
                if t_idx != -1 and t_symbol and symbols[t_idx] == " ":
                    symbols[t_idx] = t_symbol
                    _connect(symbols, idx, t_idx)

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

# 8. 출력 함수
def print_diagram():
    return f"\n{''.join(memory['characters'])}\n{''.join(memory['symbols'])}\n"

# 9. 디버깅용 테스트 함수
def test(sentence: str, verbose: bool = True):
    print("\n==============================")
    print("🟦 입력 문장:", sentence)

    store_characters(sentence)
    parsed = gpt_parse(sentence, verbose=verbose)

    print("\n📊 Parsed JSON:")
    print(json.dumps(parsed, indent=2))

    apply_symbols(parsed)

    print("\n🖨 Diagram:")
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

# 11. API 엔드포인트
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    store_characters(request.sentence)
    parsed = gpt_parse(request.sentence)
    apply_symbols(parsed)
    return {"sentence": request.sentence, "diagramming": print_diagram()}

if __name__ == "__main__":
    test("They elected him president.")
    test("She found the room clean.")
    test("He said he would go.")
    test("They will have been being called.")