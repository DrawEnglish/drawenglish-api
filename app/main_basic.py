from fastapi import FastAPI
from pydantic import BaseModel

#아래 2줄은 FastAPI에서 /openapi.json 엔드포인트 추가하기 위한 부분
from fastapi.responses import FileResponse
import os

#아래 4줄은 openai GPT 호출과 환경변수 로딩을 가능하게 하기 위한 부분
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence
    diagram = f"DrawEnglish diagram for openai: '{sentence}'"
    return {"sentence": sentence, "diagramming": diagram}

#아래는 함수는 FastAPI에서 /openapi.json 엔드포인트 추가하기 위한 부분
@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openapi.json"))
    return FileResponse(file_path, media_type="application/json")