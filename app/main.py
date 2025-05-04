from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from openai import OpenAI
from test_tools import print_parsed_roles
import os
import json

# ✅ 환경 변수 로드 및 GPT 클라이언트 초기화
load_dotenv()
client = OpenAI()  # .env 파일의 OPENAI_API_KEY를 자동 인식함

app = FastAPI()

# ✅ 메모리 구조 (문자 + 심볼)
memory = {
    "characters": [],
    "symbols": []
}

# ✅ 문법 역할 → 심볼 매핑
role_to_symbol = {
    "subject": "S",
    "verb": "○",
    "object": "□",
    "complement": "△",
    "preposition": "▽",
    "conjunction": "◇"
}

# ✅ 요청/응답 모델 정의
class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

# ✅ 문자 저장 (심볼 공간 초기화)
def store_characters(sentence: str):
    memory["characters"] = list(sentence)
    memory["symbols"] = [" " for _ in memory["characters"]]

# ✅ GPT를 통해 문장 분석
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

# ✅ /analyze 엔드포인트 - Swagger UI에서 확인 가능
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

# ✅ custom-openapi.json 제공 (GPTs용)
@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")