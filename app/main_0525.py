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
    "indirect object": "□",
    "direct object": "□",
    "prepositional object": "□",
    "preposition": "▽",
    "conjunction": "◇",
    "noun subject complement": "[",
    "adjective subject complement": "(",
    "noun object complement": "[",
    "adjective object complement": "("
}

SVOC_as_name = {
    "name", "appoint", "call", "elect", "consider", "make"
    # 일부 make도 포함, He considered her a friend.
}

noSubjectComplementVerbs = {
    "live", "arrive", "go", "come", "sleep", "die", "run", "walk", "travel", "exist", "happen"
}

noObjectVerbs = {
    "die", "arrive", "exist", "go", "come", "vanish", "fall", "sleep", "occur"
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
        t["children"] = [c["idx"] for c in tokens if c["head_idx"] == t["idx"]]

        # role 추론
        role = guess_role(t, tokens)
        if role:
            t["role"] = role  # combine에서 쓰일 수 있음

    # 'name'과 같은 동사가 있는 SVOC구조에서 목적보어를 잘못 태깅하는 것 보정 함수
    result = tokens  # 기존 tokens을 수정하며 계속 사용
    result = assign_svoc_complement_as_name(result)

    # ✅ combine 재계산
    for t in result:
        combine = guess_combine(t, result)
        if combine:
            t["combine"] = combine

    # ✅ level 재계산
    for t in result:
        t["level"] = guess_level(t, result)

    # ✅ 보어 기반 object 복구 자동 적용
    result = repair_object_from_complement(result)        

    return result


# role 추론 함수수
def guess_role(t, all_tokens=None):  # all_tokens 추가 필요
    dep = t["dep"]
    pos = t["pos"]
    head_idx = t.get("head_idx")

    # ✅ Subject
    if dep in ["nsubj", "nsubjpass"]:
        return "subject"

    # ✅ Main Verb: be동사 포함, 종속절도 고려
    if pos in ["VERB", "AUX"] and dep in ["ROOT", "ccomp", "advcl", "acl"]:
        return "verb"

    # ✅ Indirect Object
    if dep in ["iobj", "dative"]:
        return "indirect object"

    # ✅ Direct Object (SVOO 구조 판단)
    if dep in ["dobj", "obj"]:
        if head_lemma := next((t["lemma"] for t in all_tokens if t["idx"] == head_idx), None):
            if head_lemma in noObjectVerbs:
                return None  # ❌ 목적어 금지 동사 → 무시

        # ✅ 기존 object 판단 로직
        if all_tokens:
            for other in all_tokens:
                if other.get("dep") in ["iobj", "dative"] and other.get("head_idx") == head_idx:
                    return "direct object"
        return "object"

    # ✅ Preposition
    if dep == "prep":
        return "preposition"
    
    # ✅ Prepositional Object
    if dep == "pobj":
        return "prepositional object"

    # ✅ Conjunction or Clause Marker (접속사)
    if dep in ["cc", "mark"]:
        return "conjunction"

    # ✅ Subject Complement (SVC 구조)
    if dep in ["attr", "acomp"]:
        if head_lemma := next((t["lemma"] for t in all_tokens if t["idx"] == head_idx), None):
            if head_lemma in noSubjectComplementVerbs:
                return None  # ❌ 보어 불가 동사 → 차단

        if pos in ["NOUN", "PROPN", "PRON"]:
            return "noun subject complement"
        elif pos == "ADJ":
            return "adjective subject complement"

    # ✅ Object Complement (정상 케이스: dep=oprd, xcomp)
    if dep in ["oprd", "xcomp", "ccomp"]:
        if pos in ["NOUN", "PROPN", "PRON"]:
            return "noun object complement"
        elif pos == "ADJ":
            return "adjective object complement"

    # ✅ Object Complement (보완 케이스: dep=advmod, pos=ADJ, head=VERB, dobj 존재 시)
    if dep == "advmod" and pos == "ADJ":
        for tok in all_tokens:
            if tok["idx"] == head_idx and tok["pos"] == "VERB":
                obj_exists = any(
                    c.get("head_idx") == tok["idx"] and c.get("dep") in ["dobj", "obj"]
                    for c in all_tokens
                )
                if obj_exists:
                    return "adjective object complement"

    # ✅ 그 외는 DrawEnglish 도식에서 사용 안 함
    return None


def assign_svoc_complement_as_name(parsed):
    """
    SVOC 구조 동사들(SVOC_as_name 사전에 등록)의 목적보어가 spaCy에서 잘못 태깅된 경우 noun object complement로 1회 보정
    단, object 이후의 단어만 대상으로 한다.
    """
    applied = False

    for i, token in enumerate(parsed):
        if token.get("lemma") in SVOC_as_name and token.get("pos") in ["VERB", "AUX"]:
            verb_idx = token["idx"]

            # object 확인
            obj = next((t for t in parsed if t.get("head_idx") == verb_idx and t.get("dep") in ["dobj", "obj"]), None)
            if not obj:
                continue

            obj_idx = obj["idx"]

            # 보어 후보 찾기 : objedt 뒤에 있는 명사사
            for t in parsed:
                if applied:
                    break

                if (
                    t.get("idx") > obj_idx and  # ✅ object 이후에 등장한 단어만
                    t.get("head_idx") == verb_idx and
                    t.get("dep") in ["nsubj", "nmod", "attr", "appos", "npadvmod", "ccomp"] and
                    t.get("pos") in ["NOUN", "PROPN"]
                ):
                    t["role"] = "noun object complement"
                    applied = True
                    break

    return parsed


# object complement가 있는데, 앞쪽 목적어를 nsubj(subject)로 잘못 태깅하는 경우 예외처리
def repair_object_from_complement(parsed):
    for item in parsed:
        if item.get("role") in ["noun object complement", "adjective object complement"]:
            complement_children = item.get("children", [])
            for t in parsed:
                if t.get("dep") in ["nsubj", "compound"] and t.get("idx") in complement_children:
                    t["role"] = "object"
    return parsed


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

    # 1️⃣ 종속접속사 (mark): that, because, if, although, when 등
    if dep == "mark" and text in ["that", "because", "if", "although", "when", "since", "while", "unless"]:
        return 0.5

    # 2️⃣ 관계절: 관계대명사/형용사/부사 (relcl, acl 등)
    if dep in ["relcl", "acl"]:
        return 0.5

    # 3️⃣ 관계사 (관계대명사/부사) 자체
    if text in ["who", "which", "that", "where", "when", "why", "whose", "whom"]:
        return 0.5

    # 4️⃣ 복합관계사 (whoever, whatever, whichever, wherever 등)
    if text in ["whoever", "whatever", "whichever", "wherever", "whomever", "whenever", "however"]:
        return 0.5

    # 5️⃣ 의문사 (what, where, when 등) + 의문문이 아닌 문장 내부에 있을 때
    if text in ["what", "who", "which", "where", "when", "why", "how"] and dep in ["nsubj", "dobj", "pobj"]:
        return 0.5

    # 6️⃣ to부정사
    if text == "to":
        for child in t.get("children", []):
            for tok in all_tokens:
                if tok["text"] == child and tok["pos"] == "VERB":
                    return 0.5

    # 7️⃣ 현재분사 (VBG), 과거분사 (VBN)
    if tag.endswith("VBG") or tag.endswith("VBN"):
        return 0.5

    # 기본값
    return 0



# 트리거 발생시 +0.5 그 다음 단어 level +1
def propagate_levels(parsed_tokens):
    final = []
    current_level = 0  # 기본은 main level 0

    # 각 토큰에 대해 순차적으로 level 부여
    for i, token in enumerate(parsed_tokens):
        level = token.get("level", 0)

        # 트리거가 감지되면: 0.5 → 1.5 → 2.5 → ...
        if isinstance(level, float) and level % 1 == 0.5:
            token["level"] = current_level + 0.5
            current_level += 1  # 다음 절로 넘어가므로 +1
        else:
            token["level"] = current_level  # 일반 토큰은 현재 level 유지

        final.append(token)

    return final



# ◎ GPT 프롬프트 처리 함수
def spacy_parsing_backgpt(sentence: str, force_gpt: bool = False):
    doc = nlp(sentence)

    prompt = f"""

"""
    # spaCy에서 토큰 데이터 추출
    tokens = []
    for token in doc:
        morph = token.morph.to_dict()
        tokens.append({
            "idx": token.idx, "text": token.text, "pos": token.pos_,"tag": token.tag_,
            "dep": token.dep_, "head": token.head.text, "head_idx": token.head.idx,
            "tense": morph.get("Tense"), "voice": morph.get("Voice"), "form": morph.get("VerbForm"),
            "morph": morph, "lemma": token.lemma_, "is_stop": token.is_stop,
            "is_punct": token.is_punct, "is_alpha": token.is_alpha, "ent_type": token.ent_type_,
            "is_title": token.is_title, "children": [child.text for child in token.children]
        })

    # 규칙 기반 파싱
    parsed = rule_based_parse(tokens)

    # 조건: 규칙 기반 실패하거나, 강제로 GPT 사용 요청
    if not parsed or force_gpt:
        prompt = gpt_parsing_withprompt(tokens)  # 아래 2단계에서 만들 예정

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert sentence analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print("[ERROR] GPT parsing failed:", e)
            print("[RAW CONTENT]", content if 'content' in locals() else '[No response]')
            return []

    # level 보정
    parsed = propagate_levels(parsed)

    return parsed

# GPT API Parsing(with 프롬프트)을 이용하기 위한 함수
def gpt_parsing_withprompt(tokens: list) -> str:
    token_lines = []
    for t in tokens:
        token_lines.append(
            f"● idx({t['idx']}), text({t['text']}), pos({t['pos']}), tag({t['tag']}), dep({t['dep']}), head({t['head']})"
        )
    token_block = "\n".join(token_lines)

    prompt = f"""
Given the following tokenized and POS-tagged English sentence, analyze its syntactic structure.

Token Info:
{token_block}

Return a JSON list with each token's role in the sentence.
Each item must have: idx, text, role, and optionally combine/level.

If unsure, return best-guess. Do not return explanations, just the JSON.
"""
    return prompt.strip()


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
def t(sentence: str):
    parsed = spacy_parsing_backgpt(sentence)
    parsed = propagate_levels(parsed)
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

def t1(sentence: str):
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

        child_texts = [t['text'] for t in parsed if t['idx'] in token.get('children', [])]

        print(f"● idx({idx}), text({text}), role({role}), combine({combine_str}), level({level})")
        print(f"  POS({token['pos']}), DEP({token['dep']}), TAG({token['tag']}), HEAD({token['head']})")
        print(f"  lemma({token['lemma']}), is_stop({token['is_stop']}), is_punct({token['is_punct']}), is_title({token['is_title']})")
        print(f"  morph({token['morph']})")
        print(f"  children({child_texts}")
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
    "spacy_parsing_backgpt",
    "gpt_parsing_withprompt",
    "init_memorys",
    "apply_symbols",
    "symbols_to_diagram",
    "t", "t1"
]

# 테스트 문장 자동 실행


# ◎ 분석 API 엔드포인트
@app.post("/analyze", response_model=AnalyzeResponse)  # sentence를 받아 "sentence"와 "diagramming" 리턴
async def analyze(request: AnalyzeRequest):            # sentence를 받아 다음 처리로 넘김
    init_memorys(request.sentence)                     # 이 함수로 메모리 내용 채움 또는 초기화
    parsed = spacy_parsing_backgpt(request.sentence)               # GPT의 파싱결과를 parsed에 저장
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
