import os, json, re
import spacy
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse  # render에 10분 단위 Ping 보내기를 위해 추가
from pydantic import BaseModel
# 아래 api_key= 까지는 .env 파일에서 OpenAI키를 불러오기 관련 부분 
from openai import OpenAI
from dotenv import load_dotenv

# ◎ 환경 설정
load_dotenv()

api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY is not set in environment variables.")
client = OpenAI(api_key=api_key)

app = FastAPI()  # FastAPI() 객체를 생성해서 이후 라우팅에 사용
nlp = spacy.load("en_core_web_sm")  # spaCy 관련 설정, (englihs_core모델_web기반_small)

# ◎ 메모리 구조
memory = {
#    "characters": [],
    "symbols_by_level": {},
}

# ◎ 심볼 매핑
role_to_symbol = {
    "verb": "○",
    "object": "□",
    "direct object": "□",
    "indirect object": "□",
    "prepositional object": "□",
    "preposition": "▽",
    "conjunction": "◇",
    "noun subject complement": "[",
    "adjective subject complement": "(",
    "noun object complement": "[",
    "adjective object complement": "("
}

# ◎ 요청/응답 목록
class AnalyzeRequest(BaseModel):   # 사용자가 보낼 요청(sentence) 정의
    sentence: str

class AnalyzeResponse(BaseModel):  # 응답으로 돌려줄 데이터(sentence, diagramming) 정의
    sentence: str
    diagramming: str               # "     ○______□__[         "

class ParseRequest(BaseModel):     # spaCy 관련 설정
    text: str


# rule 기반 분석 뼈대 함수 선언
def rule_based_parse(tokens):
    result = []
    for t in tokens:
        item = {
            "idx": t["idx"],
            "text": t["text"],
            "level": 0  # 나중에 수정
        }
        # role 추론
        role = guess_role(t)
        if role:
            item["role"] = role
        # combine 추론
        combine = guess_combine(t, tokens)
        if combine:
            item["combine"] = combine
        result.append(item)
    return result

# role 추론 함수수
def guess_role(t):
    dep = t["dep"]
    pos = t["pos"]
    
    # ✅ Subject
    if dep in ["nsubj", "nsubjpass"]:
        return "subject"

    # ✅ Main Verb (only one per clause)
    if dep == "ROOT" and pos == "VERB":
        return "verb"

    # ✅ Direct / Indirect Object
    if dep == "iobj":
        return "indirect object"
    if dep in ["dobj", "obj"]:
        return "object"

    # ✅ Prepositional Object
    if dep == "pobj":
        return "prepositional object"

    # ✅ Preposition
    if dep == "prep":
        return "preposition"

    # ✅ Conjunction or Clause Marker (접속사)
    if dep in ["cc", "mark"]:
        return "conjunction"

    # ✅ Subject Complement (SVC 구조)
    if dep in ["attr", "acomp"]:
        if pos in ["NOUN", "PROPN", "PRON"]:
            return "noun subject complement"
        elif pos == "ADJ":
            return "adjective subject complement"

    # ✅ Object Complement (SVOC 구조)
    if dep in ["xcomp", "oprd", "ccomp"]:
        if pos in ["NOUN", "PROPN", "PRON"]:
            return "noun object complement"
        elif pos == "ADJ":
            return "adjective object complement"

    # ✅ 그 외는 DrawEnglish 도식에서 사용 안 함
    return None

# combine 추론 함수
def guess_combine(token, all_tokens):
    role = token.get("role")
    idx = token.get("idx")
    combine = []

    # ✅ Verb → object / complement (SVO, SVC)
    if role == "verb":
        for t in all_tokens:
            if t.get("head_idx") == idx:
                r = t.get("role")
                if r in [
                    "object", 
                    "indirect object", 
                    "noun subject complement", 
                    "adjective subject complement"
                ]:
                    combine.append({"text": t["text"], "role": r})

    # ✅ Indirect object → direct object (SVOO 구조)
    if role == "indirect object":
        for t in all_tokens:
            if t.get("head_idx") == idx and t.get("role") == "object":
                combine.append({"text": t["text"], "role": "direct object"})

    # ✅ Object → object complement (SVOC 구조)
    if role == "object":
        for t in all_tokens:
            if t.get("head_idx") == idx and "object complement" in (t.get("role") or ""):
                combine.append({"text": t["text"], "role": t["role"]})

    # ✅ Preposition → prepositional object
    if role == "preposition":
        for t in all_tokens:
            if t.get("head_idx") == idx and t.get("role") == "prepositional object":
                combine.append({"text": t["text"], "role": "prepositional object"})

    return combine if combine else None

# level 추론 함수수
def guess_level(t, all_tokens):
    text = t["text"].lower()
    dep = t["dep"]
    pos = t["pos"]
    tag = t["tag"]

    # 📍 1. Subordinating conjunctions (e.g., that, because)
    if dep == "mark" and text in ["that", "because", "if", "although", "since", "when", "while"]:
        return 0.5

    # 📍 2. to + verb (infinitive)
    if text == "to":
        for child in t["children"]:
            for tok in all_tokens:
                if tok["text"] == child and tok["pos"] == "VERB":
                    return 0.5

    # 📍 3. Present participle (VBG)
    if tag.endswith("VBG"):
        return 0.5

    # 📍 4. Past participle (VBN)
    if tag.endswith("VBN"):
        return 0.5

    # 📍 5. Relative / Wh-words
    if text in ["what", "who", "which", "that", "how", "why", "where", "whose", "whom", "whoever", "whatever", "whichever"]:
        return 0.5

    # ✅ 기본값
    return 0


# 0.5 level 다음 토큰들은 자동으로 n처리
def propagate_levels(parsed_tokens):
    current_level = 0
    final = []

    for i, token in enumerate(parsed_tokens):
        level = token.get("level", 0)

        if isinstance(level, float) and level % 1 == 0.5:
            current_level += 1
            token["level"] = level  # 그대로 유지 (예: 0.5)
        else:
            token["level"] = current_level

        final.append(token)

    return final
    



# ◎ GPT 프롬프트 처리 함수
def gpt_prompt_process(sentence: str):
    doc = nlp(sentence)

    prompt = f"""

"""
    # spaCy에서 토큰 데이터 추출
    tokens = []
    for token in doc:
        morph = token.morph.to_dict()
        tokens.append({
            "idx": token.idx,
            "text": token.text,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.text,
            "head_idx": token.head.idx,
            "tense": morph.get("Tense"),
            "voice": morph.get("Voice"),
            "form": morph.get("VerbForm"),
            "morph": morph,
            "lemma": token.lemma_,
            "is_stop": token.is_stop,
            "is_punct": token.is_punct,
            "is_alpha": token.is_alpha,
            "ent_type": token.ent_type_,
            "is_title": token.is_title,
            "children": [child.text for child in token.children]
        })

    # 추론 처리
    parsed = rule_based_parse(tokens)

    # level 보정
    parsed = propagate_levels(parsed)

    return parsed


# ◎ 저장공간 초기화
def init_memorys (sentence: str):
#    memory["characters"] = list(sentence)        # characters에 sentence의 글자 한글자씩 채우기
    memory["symbols_by_level"] = {}  # 문장마다 새로 초기화
    memory["sentence_length"] = len(sentence)  # 도식 길이 추적용 (줄 길이 통일)


# ◎ symbols 메모리에 심볼들 저장하기
def apply_symbols(parsed):
    symbols_by_level = memory["symbols_by_level"]
    line_length = memory["sentence_length"]

    for item in parsed:
        idx = item.get("idx", -1)
        role = item.get("role", "").lower()
        level = item.get("level")

        if idx < 0 or level is None:
            continue

        # ✅ 0.5처럼 경계 레벨은 두 줄에 심볼 찍기
        levels = [level]
        if isinstance(level, float) and level % 1 == 0.5:
            levels = [int(level), int(level) + 1]

        # 
        symbol = role_to_symbol.get(role)

        for lvl in levels:
            line = symbols_by_level.setdefault(lvl, [" " for _ in range(line_length)])
            if 0 <= idx < len(line) and line[idx] == " " and symbol:
                line[idx] = symbol


# ◎ memory["symbols"] 내용을 출력하기 위해 만든 함수
def symbols_to_diagram(sentence: str):
    """
    Returns a visual diagram (string) of current symbols_by_level in memory.
    Includes:
    - index line
    - sentence line
    - symbol lines by level
    """
    output_lines = []

    line_length = memory["sentence_length"]

    # ✅ 1. 인덱스 줄
    index_line = ''.join(str(i % 10) for i in range(line_length))
    output_lines.append(index_line)

    # ✅ 2. 문장 줄
    output_lines.append(sentence)

    # ✅ 3. 심볼 줄들 (level 순 정렬)
    for level in sorted(memory["symbols_by_level"]):
        line = ''.join(memory["symbols_by_level"][level])
        output_lines.append(line)

    return '\n'.join(output_lines)



# ◎ 디버깅용 테스트 함수
def test(sentence: str):
    parsed = gpt_prompt_process(sentence)
    print("\n📊 Parsed Result:")
    for item in parsed:
        idx = item.get("idx")
        text = item.get("text")
        role = item.get("role")
        level = item.get("level")

        # combine은 리스트거나 None
        combine = item.get("combine")
        if combine:
            combine_str = "[" + ', '.join(
                f"{c.get('text')}:{c.get('role')}" for c in combine
            ) + "]"
        else:
            combine_str = "None"

        print(f"● idx({idx}), text({text}), role({role}), combine({combine_str}), level({level})")

    print(f"\n🛠 Diagram:")
    init_memorys(sentence)
    apply_symbols(parsed)
    print(symbols_to_diagram(sentence))

def test_all(sentence: str):
    print(f"\n📘 Sentence: {sentence}")
    doc = nlp(sentence)
    morph_data = []  # 전체 토큰 리스트 저장

    # spaCy에서 full 토큰 추출
    for token in doc:
        morph = token.morph.to_dict()
        morph_data.append({
            "idx": token.idx, "text": token.text, "pos": token.pos_,
            "tag": token.tag_, "dep": token.dep_, "head": token.head.text,
            "head_idx": token.head.idx, "tense": morph.get("Tense"),
            "form": morph.get("VerbForm"), "voice": morph.get("Voice"),
            "morph": morph, "lemma": token.lemma_,
            "is_stop": token.is_stop, "is_punct": token.is_punct, "is_alpha": token.is_alpha,
            "ent_type": token.ent_type_, "is_title": token.is_title,
            "children": [child.text for child in token.children]
        })

    # 구조 추론
    parsed = rule_based_parse(morph_data)
    parsed = propagate_levels(parsed)

    print("\n📊 Full Token Info with Annotations:")
    for token in morph_data:
        idx = token["idx"]
        text = token["text"]
        role = next((t.get("role") for t in parsed if t["idx"] == idx), None)
        combine = next((t.get("combine") for t in parsed if t["idx"] == idx), None)
        level = next((t.get("level") for t in parsed if t["idx"] == idx), None)

        combine_str = (
            "[" + ", ".join(f"{c['text']}:{c['role']}" for c in combine) + "]"
            if combine else "None"
        )

        print(f"● idx({idx}), text({text}), role({role}), combine({combine_str}), level({level})")
        print(f"  POS({token['pos']}), TAG({token['tag']}), DEP({token['dep']}), HEAD({token['head']})")
        print(f"  lemma({token['lemma']}), is_stop({token['is_stop']}), is_punct({token['is_punct']}), is_title({token['is_title']})")
        print(f"  morph({token['morph']})")
        print(f"  children({token['children']})")
        print("")

    # 도식 출력
    print("🛠 Diagram:")
    init_memorys(sentence)
    apply_symbols(parsed)
    print(symbols_to_diagram(sentence))



# 초간단 임시 테스트1 함수


# ◎ 모듈 외부 사용을 위한 export
__all__ = [
    "rule_based_parse",
    "guess_role",
    "guess_combine",
    "guess_level",
    "propagate_levels",
    "gpt_prompt_process",
    "init_memorys",
    "apply_symbols",
    "symbols_to_diagram",
    "test"
]

# 테스트 문장 자동 실행
#if __name__ == "__main__":
#    test("He told her that she is smart.")


# ◎ 분석 API 엔드포인트
@app.post("/analyze", response_model=AnalyzeResponse)  # sentence를 받아 "sentence"와 "diagramming" 리턴
async def analyze(request: AnalyzeRequest):            # sentence를 받아 다음 처리로 넘김
    init_memorys(request.sentence)                     # 이 함수로 메모리 내용 채움 또는 초기화
    parsed = gpt_prompt_process(request.sentence)               # GPT의 파싱결과를 parsed에 저장
    apply_symbols(parsed)                              # parsed 결과에 따라 심볼들을 메모리에 저장장
    return {"sentence": request.sentence,
            "diagramming": symbols_to_diagram(request.sentence)}

# ◎ spaCy 파싱 관련
@app.post("/parse")
def parse_text(req: ParseRequest):
    doc = nlp(req.text)
    result = []

    for token in doc:
        morph = token.morph.to_dict()
        result.append({
            "idx": token.idx, "text": token.text, "pos": token.pos_, "tag": token.tag_,
            "dep": token.dep_, "head": token.head.text, "head_idx": token.head.idx,
            "tense": morph.get("Tense"), "form": morph.get("VerbForm"),
            "voice": morph.get("Voice"), "morph": morph, "lemma": token.lemma_,
            "is_stop": token.is_stop, "is_punct": token.is_punct, "is_alpha": token.is_alpha,
            "ent_type": token.ent_type_, "is_title": token.is_title,
            "children": [child.text for child in token.children]
        })

    return {"result": result}

# ◎ 커스텀 OpenAPI JSON 제공 엔드포인트
# FastAPI에서 custom-openapi.json 엔드포인트를 만들어서 GPTs에서 사용할 수 있도록 함.
@app.get("/custom-openapi.json", include_in_schema=False)
async def serve_openapi():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openapi.json"))
    # file_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    return FileResponse(file_path, media_type="application/json")

# ◎ 아래 엔드포인트는 GET /ping 요청에 대해 {"message": "pong"} 응답을 준다.
@app.get("/ping")
async def ping():
    return JSONResponse(content={"message": "pong"}, status_code=200)
#
