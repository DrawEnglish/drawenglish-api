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

## 메모리 (문자 + 심볼)
memory = {
    "characters": [],
    "symbols": []
}

class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

## 문법 요소 → 심볼 매핑
role_to_symbol = {
    "subject": "S",
    "verb": "○",
    "object": "□",
    "complement": "△",
    "preposition": "▽",
    "conjunction": "◇"
}

## 문자 저장
def store_characters(sentence: str):
    memory["characters"] = list(sentence)
    memory["symbols"] = [" " for _ in memory["characters"]]  # 초기화

## GPT로 문장 분석 요청
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

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert English grammar analyzer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0  # 결과 일관성 위해
    )

    content = response['choices'][0]['message']['content']
    
    try:
        parsed_result = eval(content)  # GPT가 JSON 포맷 준다고 가정 (주의: 실제 서비스는 json.loads() 추천)
        return parsed_result
    except:
        return []

## 심볼 적용
def apply_symbols(parsed_result):
    text = ''.join(memory["characters"]).lower()

    for item in parsed_result:
        word = item["word"].lower()
        role = item["role"].lower()
        symbol = role_to_symbol.get(role)

        if symbol:
            index = text.find(word)
            if index != -1:
                memory["symbols"][index] = symbol

## 다이어그램 출력
def generate_diagram():
    output = ""
    for char, symbol in zip(memory["characters"], memory["symbols"]):
        if symbol != " ":
            output += f"{symbol}{char}"
        else:
            output += f" {char}"
    return output

## 엔드포인트
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence

    # 1. 문자 저장
    store_characters(sentence)

    # 2. GPT 분석 호출
    parsed = gpt_parse(sentence)

    # 3. 심볼 적용
    apply_symbols(parsed)

    # 4. 다이어그램 생성
    diagram = generate_diagram()

    return {
        "sentence": sentence,
        "diagramming": diagram
    }


# @app.post("/analyze", response_model=AnalyzeResponse)
# async def analyze(request: AnalyzeRequest):
#    sentence = request.sentence
#    diagram = f"DrawEnglish diagram for openai: '{sentence}'"
#    return {"sentence": sentence, "diagramming": diagram}

#아래는 함수는 FastAPI에서 /openapi.json 엔드포인트 추가하기 위한 부분
@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")