FROM python:3.11-slim

# 작업 디렉토리 설정
# WORKDIR /code
WORKDIR /app

# 애플리케이션 코드 복사
COPY . .

# 의존성 설치
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

ENV PYTHONUNBUFFERED=1

# FastAPI 실행
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
CMD ["python", "app/main.py"]
