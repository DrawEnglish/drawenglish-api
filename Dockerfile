# 1. 경량 Python 베이스 이미지
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 복사
COPY requirements.txt .

# 4. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. spaCy 모델 설치 (환경변수 SPACY_MODEL에 따라)
RUN python -c "import os; import spacy; model = os.getenv('SPACY_MODEL', 'en_core_web_sm'); spacy.cli.download(model)"

# 6. 앱 코드 복사
COPY . .

# 7. 로그 실시간 출력되도록
ENV PYTHONUNBUFFERED=1

# 8. FastAPI 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]

