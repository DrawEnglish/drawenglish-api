# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os

# 🔽 분리된 모듈 불러오기
from analyzer_utils import gpt_parse, store_characters
from subcode import apply_symbols, generate_diagram

app = FastAPI()

class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence

    # 1. 문자 저장
    store_characters(sentence)

    # 2. GPT 분석 호출
    parsed_result = gpt_parse(sentence)

    if not parsed_result:
        diagram = "❌ Parsing failed."
    else:
        # 3. 심볼 적용 및 다이어그램 생성
        apply_symbols(parsed_result)
        diagram = generate_diagram()

    return {"sentence": sentence, "diagramming": diagram}

@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "openapi.json")
    return FileResponse(file_path, media_type="application/json")
