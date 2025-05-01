from fastapi import FastAPI
from pydantic import BaseModel

#아래는 2줄은 FastAPI에서 /openapi.json 엔드포인트 추가하기 위한 부분
from fastapi.responses import FileResponse
import os

app = FastAPI()

class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence
    diagram = f"DrawEnglish diagram for: '{sentence}'"
    return {"sentence": sentence, "diagramming": diagram}

#아래는 함수는 FastAPI에서 /openapi.json 엔드포인트 추가하기 위한 부분
@app.get("/openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")