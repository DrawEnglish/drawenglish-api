# analyzer_utils.py

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ✅ 메모리 구조 (공유됨)
memory = {
    "characters": [],
    "symbols": []
}

# ✅ 문법 요소 → 심볼 매핑
role_to_symbol = {
    "subject": "S",
    "verb": "○",
    "object": "□",
    "complement": "△",
    "preposition": "▽",
    "conjunction": "◇"
}

def store_characters(sentence: str):
    memory["characters"] = list(sentence)
    memory["symbols"] = [" " for _ in memory["characters"]]

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

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert English grammar analyzer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return []

def print_parsed_roles(sentence: str):
    parsed_result = gpt_parse(sentence)

    if not parsed_result:
        print("❌ GPT 응답을 파싱하지 못했습니다.")
        return

    for item in parsed_result:
        word = item.get("word")
        role = item.get("role")
        print(f"{word} - {role}")
