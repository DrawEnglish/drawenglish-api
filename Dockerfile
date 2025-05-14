# 1. 경량 Python 베이스 이미지
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 먼저 복사하고 설치 (Docker 캐시 최적화)
COPY requirements.txt .

# 4. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. spaCy 모델 설치 (미리 다운받아두기)
RUN python -m spacy download en_core_web_sm

# 6. 전체 소스 코드 복사
COPY . .

# 7. 로그 실시간 출력되도록
ENV PYTHONUNBUFFERED=1

# 8. FastAPI 앱 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]