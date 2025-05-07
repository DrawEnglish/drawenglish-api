FROM python:3.11-slim

# 1. 작업 디렉토리 설정
WORKDIR /code

# 2. 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. .env 복사 (.gitignore에 등록된 상태면 GitHub에 푸시되지 않음)
COPY env.prod .env

# 4. 애플리케이션 코드 복사
COPY . .

# 5. FastAPI 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
