from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from openai import OpenAI
import os
import json

# --------------------------------------------------
# 1. 환경 초기화
# --------------------------------------------------
load_dotenv()
client = OpenAI()
app = FastAPI()

# --------------------------------------------------
# 2. 메모리 구조 선언
# --------------------------------------------------
memory = {
    "characters": [],
    "symbols": []
}

# --------------------------------------------------
# 3. 문법 역할 → 심볼 매핑 (subject 제외)
# --------------------------------------------------
role_to_symbol = {
    "verb": "○",
    "object": "□",
    "noun complement": "[",
    "adjective complement": "(",
    "preposition": "▽",
    "conjunction": "◇"
}

# --------------------------------------------------
# 4. 요청/응답 모델
# --------------------------------------------------
class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

# --------------------------------------------------
# 5. 문자 저장 + 초기화
# --------------------------------------------------
def store_characters(sentence: str):
    memory["characters"] = list(sentence)
    memory["symbols"] = [" " for _ in memory["characters"]]

# --------------------------------------------------
# 6. GPT 문장 분석 + 프롬프트 강화
# --------------------------------------------------
def gpt_parse(sentence: str):
    prompt = f"""
Analyze the following English sentence.

For each **meaningful word** (excluding punctuation), identify its grammatical role using **only** the following labels:
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

2. If the word is a direct or indirect object of a verb (e.g., "They offered us a job"), label it as 'object', **not** as a complement.

3. Do **not** classify determiners like "a", "an", or "the" as prepositions.

4. Do not include punctuation marks in the result.

### Examples:

- He is a teacher. → 'a teacher' = noun complement
- They elected him president. → 'president' = noun complement
- They offered us a job. → 'job' = object ✅
- The dog chased the cat. → 'dog' = subject, 'chased' = verb, 'cat' = object

Return the result as a JSON array. Each item should be an object with exactly two fields: "word" and "role". Do not include any explanations.

Example:
[
  {{"word": "I", "role": "subject"}},
  {{"word": "love", "role": "verb"}},
  {{"word": "you", "role": "object"}}
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



# --------------------------------------------------
# 7. GPT 결과 기반 심볼 적용 + 후처리 검증
# --------------------------------------------------
def apply_symbols(parsed_result):
    char_join = ''.join(memory["characters"]).lower()

    for item in parsed_result:
        word = item.get("word", "").lower()
        role = item.get("role", "").lower()

        # ❗ 후처리: 잘못된 preposition 제거
        if role == "preposition" and word in ["a", "an", "the"]:
            continue

        symbol = role_to_symbol.get(role, " ")
        if not symbol.strip():
            continue

        index = char_join.find(word)
        if index != -1:
            memory["symbols"][index] = symbol

# --------------------------------------------------
# 8. 다이어그램 생성
# --------------------------------------------------
def print_diagrams():
    char_line = "".join(memory["characters"])
    symb_line = "".join(memory["symbols"])
    return f"{char_line}\n{symb_line}"

# --------------------------------------------------
# 9. FastAPI /analyze 엔드포인트
# --------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence
    store_characters(sentence)
    parsed_result = gpt_parse(sentence)
    apply_symbols(parsed_result)
    diagrams = print_diagrams()

    return {"sentence": sentence, "diagramming": diagrams}

# --------------------------------------------------
# 10. GPTs용 custom-openapi.json 제공
# --------------------------------------------------
@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")

# --------------------------------------------------
# 11. 콘솔 테스트 함수 (추가됨!)
# --------------------------------------------------
def parse_test(sentence: str):
    print("\n🟦 입력 문장:", sentence)
    
    # 1. 문자 저장
    store_characters(sentence)

    # 2. GPT 분석 결과
    parsed = gpt_parse(sentence)

    # 3. 분석 결과 출력
    print("\n[🔍 Parsed Result]")
    for item in parsed:
        word = item.get("word", "")
        role = item.get("role", "")
        print(f"- {word}: {role}")

    # 4. 심볼 적용 + 다이어그램 출력
    apply_symbols(parsed)
    print("\n[🖨 Diagrams]")
    print(print_diagrams())
    import json
    print(json.dumps(parsed, indent=2))

    # return parsed  # 원하면 외부에서 쓸 수 있도록 반환

# 콘솔 실행용
if __name__ == "__main__":
    parse_test("I give him a book.")
    parse_test("The weather is beautiful.")
