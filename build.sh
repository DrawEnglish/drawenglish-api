#!/bin/bash

# Render 배포용 requirements 복사
cp requirements_render.txt requirements.txt

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
