from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import re  # 정규표현식

# 1. 환경 초기화
load_dotenv()
client = OpenAI()
app = FastAPI()

# 2. 메모리 구조
memory = {
    "characters": [],
    "symbols": []
}

# 3. 심볼 매핑
role_to_symbol = {
    "verb": "○",
    "object": "□",
    "noun complement": "[",
    "adjective complement": "(",
    "preposition": "▽",
    "conjunction": "◇"
}

# 4. 요청/응답 모델
class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

# 5. 문자 저장
def store_characters(sentence: str):
    memory["characters"] = list(sentence)
    memory["symbols"] = [" " for _ in memory["characters"]]

# 6. GPT 파싱
def gpt_parse(sentence: str):
    prompt = f"""
Analyze the following English sentence.

For each meaningful word (excluding punctuation), identify its grammatical role using only the following labels:
- subject
- verb
- object
- noun complement
- adjective complement
- preposition
- conjunction

### Instructions:

1. Use 'noun complement' or 'adjective complement' **only** when the word describes:
   - the subject after a linking verb (e.g., "He is a teacher"), or
   - the object in an SVOC structure (e.g., "They elected him president").

2. If the word is a direct or indirect object of a verb (e.g., "They offered us a job"), label it as 'object', not as a complement.

3. Do not classify "a", "an", "the" as prepositions or objects.

4. Do not include punctuation marks.

Return a JSON array:
[
  {{"word": "I", "role": "subject"}},
  ...
]

Sentence: "{sentence}"
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert English grammar analyzer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return []

# 7. 심볼 적용
def apply_symbols(parsed_result):
    char_line = ''.join(memory["characters"])
    char_lower = char_line.lower()
    symbol_line = memory["symbols"]
    word_positions = [(m.group(), m.start()) for m in re.finditer(r'\b\w+\b', char_lower)]
    used_indices = set()

    # 마지막 verb만 표시 (조동사 제거용)
    last_verb = None
    for item in reversed(parsed_result):
        if item.get("role", "").lower() == "verb":
            last_verb = item.get("word", "").lower()
            break

    for item in parsed_result:
        word = item.get("word", "").lower()
        role = item.get("role", "").lower()

        if word in ["a", "an", "the"]:
            continue
        if role == "verb" and word != last_verb:
            continue

        symbol = role_to_symbol.get(role, "")
        if not symbol.strip():
            continue

        for token, idx in word_positions:
            if token == word and idx not in used_indices:
                symbol_line[idx] = symbol
                used_indices.add(idx)
                break

# 8. 연결선 추가
def connect_symbols(parsed_result):
    char_line = ''.join(memory["characters"]).lower()
    positions = []

    for item in parsed_result:
        word = item.get("word", "").lower()
        role = item.get("role", "").lower()
        symbol = role_to_symbol.get(role, "")

        for m in re.finditer(rf'\b{re.escape(word)}\b', char_line):
            idx = m.start()
            if memory["symbols"][idx] == symbol:
                positions.append({"role": role, "index": idx})
                break

    for i in range(len(positions) - 1):
        cur = positions[i]
        nxt = positions[i + 1]
        if (cur["role"] == "verb" and nxt["role"] in ["object", "noun complement", "adjective complement"]) or \
           (cur["role"] == "object" and nxt["role"] in ["noun complement", "adjective complement"]) or \
           (cur["role"] == "preposition" and nxt["role"] == "object") or \
           (cur["role"] == "verb" and nxt["role"] == "verb"):

            for j in range(cur["index"] + 1, nxt["index"]):
                if memory["symbols"][j] == " ":
                    memory["symbols"][j] = "_"

# 9. 다이어그램 출력
def print_diagrams():
    return f"{''.join(memory['characters'])}\n{''.join(memory['symbols'])}"

# 10. FastAPI 엔드포인트
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence
    store_characters(sentence)
    parsed = gpt_parse(sentence)
    apply_symbols(parsed)
    connect_symbols(parsed)
    diagram = print_diagrams()
    return {"sentence": sentence, "diagramming": diagram}

@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")

# 11. 콘솔 테스트
def parse_test(sentence: str):
    print("\n🟦 입력 문장:", sentence)
    store_characters(sentence)
    parsed = gpt_parse(sentence)
    print("\n[🔍 Parsed Result]")
    for item in parsed:
        print(f"- {item.get('word')}: {item.get('role')}")
    apply_symbols(parsed)
    connect_symbols(parsed)
    print("\n[🖨 Diagram]")
    print(print_diagrams())
    print("\n[📦 JSON]")
    print(json.dumps(parsed, indent=2))

if __name__ == "__main__":
    parse_test("They elected him president.")
    parse_test("She found the room clean.")
    parse_test("He said he would go.")
    parse_test("They will have been being called.")
