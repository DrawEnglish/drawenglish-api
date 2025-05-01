from fastapi import FastAPI
from pydantic import BaseModel

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
