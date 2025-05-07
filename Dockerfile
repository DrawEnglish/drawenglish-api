FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# .env 파일 복사 (이미 하고 있었을 것)
COPY env.prod .env

# 👇 이 줄을 추가!
# .env의 내용을 실제 환경 변수로 등록
RUN export $(cat .env | xargs)

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
