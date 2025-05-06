from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import re  # 정규표현식

# 1. 환경 초기화
load_dotenv()
client = OpenAI()
app = FastAPI()

# 2. 메모리 구조
memory = {
    "characters": [],
    "symbols": []
}

# 3. 심볼 매핑
role_to_symbol = {
    "verb": "○",
    "object": "□",
    "noun complement": "[",
    "adjective complement": "(",
    "preposition": "▽",
    "conjunction": "◇"
}

# 4. 요청/응답 모델
class AnalyzeRequest(BaseModel):
    sentence: str

class AnalyzeResponse(BaseModel):
    sentence: str
    diagramming: str

# 5. 문자 저장
from copy import deepcopy

def store_characters(sentence: str):
    memory["characters"] = list(sentence)
    memory["symbols"] = [" " for _ in memory["characters"]]

    char_line = sentence.lower()
    word_positions = []
    for m in re.finditer(r'\b\w+\b', char_line):
        word_positions.append({
            "token": m.group(),
            "index": m.start(),
            "used": False
        })

    # 저장
    memory["word_positions_base"] = word_positions  # 🔸 이건 디버깅용
    memory["word_positions_for_symbols"] = deepcopy(word_positions)
    memory["word_positions_for_links"] = deepcopy(word_positions)


# 6. GPT 파싱
def gpt_parse(sentence: str):
    prompt = f"""
Analyze the following English sentence.

For each meaningful word (excluding punctuation), identify its grammatical role using only the following labels:
- subject
- verb
- object
- noun complement
- adjective complement
- preposition
- conjunction

### Rules:

1. One main verb per clause:
   - Each clause should have at most one word labeled as 'verb'.
   - In compound sentences joined by a conjunction (e.g. "He ran and fell"), only the first verb in the clause should be labeled 'verb'.

2. If a grammatical function spans multiple words, only tag the **core word**:
   - e.g. "my friend" → only 'friend' with role 'noun complement'
   - e.g. "very clean" → only 'clean' with role 'adjective complement'

3. Ignore:
   - Articles: 'a', 'an', 'the'
   - Possessives: 'my', 'your', etc.
   - Modifiers and adverbs like 'very', 'quickly'

4. For any 'noun complement' or 'adjective complement', 
   also include a `"target"` field to indicate whether it completes the **subject** or the **object**.
   Use: `"target": "subject"` or `"target": "object"`

5. Prepositional Phrases:
   - If a noun follows a preposition and acts as its object, label it as 'object'.
   - Do NOT label it as 'noun complement' or anything else.
   - Example: In "under the table", 'table' → role: 'object'

### Examples:

Sentence: "He is a teacher."

[
  {{ "word": "He", "role": "subject" }},
  {{ "word": "is", "role": "verb" }},
  {{ "word": "teacher", "role": "noun complement", "target": "subject" }}
]

Sentence: "They elected him president."

[
  {{ "word": "They", "role": "subject" }},
  {{ "word": "elected", "role": "verb" }},
  {{ "word": "him", "role": "object" }},
  {{ "word": "president", "role": "noun complement", "target": "object" }}
]

Sentence: "She sat on the chair."

[
  {{ "word": "She", "role": "subject" }},
  {{ "word": "sat", "role": "verb" }},
  {{ "word": "on", "role": "preposition" }},
  {{ "word": "chair", "role": "object" }}
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
        parsed = json.loads(content)
        return [item for item in parsed if "word" in item and "role" in item]
    except json.JSONDecodeError:
        return []


# 7. 심볼 적용
def apply_symbols(parsed_result):
    char_line = ''.join(memory["characters"])
    char_lower = char_line.lower()
    symbol_line = memory["symbols"]

    # 단어 위치 추적
    word_positions = memory["word_positions_for_symbols"]
    for m in re.finditer(r'\b\w+\b', char_lower):
        word_positions.append({
            "token": m.group(),
            "index": m.start(),
            "used": False
        })

    # 무시할 단어
    skip_words = ["a", "an", "the", "my", "your", "his", "her", "their", "our", "its", "very", "too", "also"]

    # 역할별로 마지막 항목만 기록
    last_indices = {}
    for i, item in enumerate(parsed_result):
        role = item.get("role", "").lower()
        if role in role_to_symbol:
            last_indices[role] = i

    # 동사 그룹 처리 (조동사 + 본동사 → 마지막만 표시)
    verb_indices = [i for i, item in enumerate(parsed_result) if item.get("role", "").lower() == "verb"]
    verb_groups = set()
    if verb_indices:
        start = verb_indices[0]
        for i in range(1, len(verb_indices)):
            if verb_indices[i] == verb_indices[i - 1] + 1:
                continue
            verb_groups.add(verb_indices[i - 1])
            start = verb_indices[i]
        verb_groups.add(verb_indices[-1])

    # 심볼 적용
    for i, item in enumerate(parsed_result):
        word = item.get("word", "").lower()
        role = item.get("role", "").lower()

        if word in skip_words:
            continue

        # verb는 그룹의 마지막 것만 허용
        if role == "verb" and i not in verb_groups:
            continue

        # 나머지 역할은 가장 마지막 항목만 허용
        if role != "verb" and i != last_indices.get(role):
            continue

        symbol = role_to_symbol.get(role, "")
        if not symbol.strip():
            continue

        for pos in word_positions:
            if pos["token"] == word and not pos["used"]:
                symbol_line[pos["index"]] = symbol
                pos["used"] = True
                break


# 8. 연결선 추가
def connect_symbols(parsed_result):
    symbol_line = memory["symbols"]
    word_positions = memory["word_positions_for_links"]
    positions = []

    connection_logs = []  # 연결 디버깅용 로그

    # 역할별 위치 추출
    for item in parsed_result:
        word = item.get("word", "").lower()
        role = item.get("role", "").lower()
        symbol = role_to_symbol.get(role, "")

        for pos in word_positions:
            if not pos["used"] and pos["token"] == word and symbol_line[pos["index"]] == symbol:
                positions.append({
                    "role": role,
                    "index": pos["index"]
                })
                pos["used"] = True
                break

    # 순서대로 연결 관계 판단
    for i in range(len(positions) - 1):
        cur = positions[i]
        nxt = positions[i + 1]

        # 금지: verb → verb
        if cur["role"] == "verb" and nxt["role"] == "verb":
            continue

        # 허용된 연결쌍만 처리
        allowed = (
            (cur["role"] == "verb" and nxt["role"] in ["object", "noun complement", "adjective complement"]) or
            (cur["role"] == "object" and nxt["role"] in ["noun complement", "adjective complement"]) or
            (cur["role"] == "preposition" and nxt["role"] == "object")
        )
        if not allowed:
            continue

        # 연결 전에 해당 구간에 이미 밑줄이 있는지 확인
        already_connected = any(symbol_line[j] == "_" for j in range(cur["index"] + 1, nxt["index"]))
        if already_connected:
            continue

        # 연결선 그리기
        for j in range(cur["index"] + 1, nxt["index"]):
            if symbol_line[j] == " ":
                symbol_line[j] = "_"

        # 로그 기록
        connection_logs.append({
            "from": cur["index"],
            "to": nxt["index"],
            "role_from": cur["role"],
            "role_to": nxt["role"],
            "span": (cur["index"] + 1, nxt["index"] - 1)
        })

    return connection_logs


# 9. 다이어그램 출력
def print_diagrams_for_console():
    char_line = ''.join(memory["characters"])
    symbol_line = ''.join(memory["symbols"])
    return f"\n{char_line}\n{symbol_line}"

def print_diagrams():
    return f"```\n{''.join(memory['characters'])}\n{''.join(memory['symbols'])}\n```"

# 10. FastAPI 엔드포인트
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    sentence = request.sentence
    store_characters(sentence)
    parsed = gpt_parse(sentence)
    apply_symbols(parsed)
    connect_symbols(parsed)
    diagram = print_diagrams()
    return {"sentence": sentence, "diagramming": diagram}

@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")

# 11. 심볼, 밑줄 그리기 디버깅 함수
def debug_symbol_positions(parsed_result, connection_logs=None):
    char_line = ''.join(memory["characters"])
    symbol_line = ''.join(memory["symbols"])
    index_line = ''.join([str(i % 10) for i in range(len(char_line))])

    print("\n🧪 [DEBUG] 문자 인덱스:")
    print(index_line)
    print(char_line)
    print(symbol_line)

    print("\n🔍 [DEBUG] 단어별 위치 및 역할:")
    word_positions = memory.get("word_positions_base", [])
    for item in parsed_result:
        word = item.get("word", "")
        role = item.get("role", "")
        idxs = [p["index"] for p in word_positions if p["token"] == word]
        print(f"- {word:10s} → {role:20s} at positions {idxs}")

    if connection_logs:
        print("\n🔗 [DEBUG] 연결된 밑줄:")
        for log in connection_logs:
            print(f"- {log['role_from']:10s} → {log['role_to']:20s} at {log['span']}")


# 12. 콘솔 테스트
def parse_test(sentence: str):
    print("\n==============================")
    print("🟦 입력 문장:", sentence)

    store_characters(sentence)
    parsed = gpt_parse(sentence)

    print("\n📊 Parsed JSON:")
    print(json.dumps(parsed, indent=2))

    apply_symbols(parsed)
    connection_logs = connect_symbols(parsed)

    print("\n🖨 Diagram:")
    print(print_diagrams_for_console())
    print("==============================\n")

    debug_symbol_positions(parsed, connection_logs)


if __name__ == "__main__":
    parse_test("They elected him president.")
    parse_test("She found the room clean.")
    parse_test("He said he would go.")
    parse_test("They will have been being called.")
