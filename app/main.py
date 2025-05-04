# ---------------------------------------------------
# 1. ✅ 환경 초기화 및 GPT 클라이언트 설정
# ---------------------------------------------------
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()  # .env에 저장된 OPENAI_API_KEY 자동 사용

# ---------------------------------------------------
# 2. ✅ FastAPI 객체 생성
# ---------------------------------------------------
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

# ---------------------------------------------------
# 3. ✅ 요청/응답 모델 정의 (Pydantic)
# ---------------------------------------------------
class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

# ---------------------------------------------------
# 4. ✅ GPT 분석 및 처리 함수
# ---------------------------------------------------

# ⬅️ 메모리 구조 (문자 + 심볼)
memory = {
    "characters": [],
    "symbols": []
}

# ⬅️ 역할 → 심볼 매핑
role_to_symbol = {
    "subject": "S",
    "verb": "○",
    "object": "□",
    "complement": "△",
    "preposition": "▽",
    "conjunction": "◇"
}

# ⬅️ 문자 저장 함수
def store_characters(sentence: str):
    memory["characters"] = list(sentence)
    memory["symbols"] = [" " for _ in memory["characters"]]

# ⬅️ GPT 호출 함수
def gpt_parse(sentence: str):
    prompt = f"""
Analyze the following English sentence.

For each meaningful word (excluding punctuation), identify its grammatical role: 
subject, verb, object, complement, preposition, or conjunction.

Return the result as a JSON array with this format:
[
  {{"word": "word", "role": "subject"}},
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

# ---------------------------------------------------
# 5. ✅ 디버깅용: 분석 결과 콘솔 출력 함수
# ---------------------------------------------------
def print_parsed_roles(sentence: str):
    parsed_result = gpt_parse(sentence)

    if not parsed_result:
        print("❌ GPT 응답을 파싱하지 못했습니다.")
        return

    for item in parsed_result:
        word = item.get("word")
        role = item.get("role")
        print(f"{word} - {role}")

# 단독 실행 시 테스트
if __name__ == "__main__":
    print_parsed_roles("I love you.")
    print_parsed_roles("She gave me a book.")

# ---------------------------------------------------
# 6. ✅ FastAPI 라우터 함수 정의
# ---------------------------------------------------

# ⬅️ POST /analyze - 문장 분석 결과 반환
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence
    parsed_result = gpt_parse(sentence)

    if not parsed_result:
        diagram = "❌ Parsing failed."
    else:
        diagram_lines = [f"{item['word']} - {item['role']}" for item in parsed_result]
        diagram = "\n".join(diagram_lines)

    return {"sentence": sentence, "diagramming": diagram}

# ⬅️ GET /custom-openapi.json - GPTs용 OpenAPI 제공
@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")
