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

# 📘 보어가 **명사만** 가능한 SVOC 동사
SVOC_noun_only = {
    "name", "appoint", "elect", "dub", "label", "christen", "nominate"
}

# 📙 보어가 **형용사만** 가능한 SVOC 동사
SVOC_adj_only = {
    "find", "keep", "leave", "consider"  # 일부 보어 명사도 가능하긴 하지만 거의 형용사 우위
}

# 📗 보어가 **명사/형용사 둘 다** 가능한 SVOC 동사
SVOC_both = {
    "make", "call", "consider", "declare", "paint", "think", "judge"
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
    verb_attribute: dict

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
    result = assign_noun_complement_for_SVOC_noun_only(result)

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
    if pos in ["VERB", "AUX"] and dep in ["ROOT", "ccomp", "advcl", "acl", "relcl"]:
        return "verb"

    # ✅ 등위접속사 다음 병렬 동사 (conj)도 verb role 부여
    if pos == "VERB" and dep == "conj":
        if any(t["idx"] == head_idx and t.get("role") == "verb" for t in all_tokens):
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

# dep가 dative여서 indiret object가 있는데, 뒤쪽에 direct object role이 없는 경우 보정
def recover_direct_object_from_indirect(parsed):
    """
    SVOO 문장에서 indirect object에 대해 appos 구조의 direct object를 복원
    """
    for token in parsed:
        if token.get("role") == "indirect object":
            token_idx = token.get("idx")

            for child in parsed:
                if (
                    child.get("head_idx") == token_idx and
                    child.get("dep") == "appos" and
                    child.get("pos") in {"NOUN", "PROPN"}
                ):
                    child["role"] = "direct object"
                    break

    return parsed


# 목적보어로 명사만 취하는 동사 사용 문장에서 목적보어를 잘못 태깅하는 것 보정
# 그 후 아래 repair_object_from_complement()함수를 통해 목적어를 보정함
def assign_noun_complement_for_SVOC_noun_only(parsed):
    """
    SVOC 구조 동사들(SVOC_noun_only 사전에 등록)의 목적보어가 spaCy에서 잘못 태깅된 경우
    noun object complement로 1회 보정 단, object 이후의 단어만 대상으로 한다.
    """
    applied = False

    for i, token in enumerate(parsed):
        if token.get("lemma") in SVOC_noun_only and token.get("pos") in ["VERB", "AUX"]:
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

# 목적보어로 형용사만 또는 형용사/명사를 모두 취하는 동사 사용 문장에서 목적보어를 잘못 태깅하는 것 보정
# 그 후 아래 repair_object_from_complement()함수를 통해 목적어를 보정함
# 예: "She painted the wall green."
def assign_adj_object_complement_when_compound_object(parsed):
    for verb in parsed:
        if verb.get("pos") != "VERB":
            continue

        verb_lemma = verb.get("lemma")
        if verb_lemma not in SVOC_adj_only and verb_lemma not in SVOC_both:
            continue

        verb_idx = verb["idx"]

        # 보어 후보: VERB의 자식 중 dep=obj, pos=ADJ
        for t in parsed:
            if (
                t.get("head_idx") == verb_idx and
                t.get("dep") in ["dobj", "obj"] and
                t.get("pos") == "ADJ"
            ):
                # ✅ 이 ADJ의 children에 compound가 붙어 있으면 보정 대상
                children = [c for c in parsed if c.get("head_idx") == t["idx"]]
                has_compound = any(
                    c.get("dep") == "compound" and c.get("pos") == "NOUN"
                    for c in children
                )
                if has_compound:
                    t["role"] = "adjective object complement"

    return parsed

# SVOC_both 동사에 속해 있고, 뒤에 role(object)가 있고, 그 뒤에 advcl, ADJ 이면서 HEAD가 object와 같을때
# adjective object complement로 보정.  예: He painted the kitchen walls blue.
def assign_adj_complement_for_advcl_adjective(parsed):
    """
    spaCy가 형용사 목적보어를 advcl로 잘못 태깅했을 때 보정
    예: "He painted the walls blue."
    """
    for verb in parsed:
        if verb.get("pos") != "VERB":
            continue
        if verb.get("lemma") not in SVOC_both:
            continue

        verb_idx = verb["idx"]

        # 1. object 있는지 먼저 확인
        obj = next(
            (t for t in parsed if t.get("head_idx") == verb_idx and t.get("role") in ["object", "direct object"]),
            None
        )
        if not obj:
            continue

        obj_idx = obj["idx"]

        # 2. object 이후 등장한 형용사 중 특정 조건을 만족하면 보어로 간주
        for t in parsed:
            if (
                t.get("idx") > obj_idx and
                t.get("head_idx") == verb_idx and
                t.get("dep") == "advcl" and
                t.get("pos") == "ADJ"
            ):
                t["role"] = "adjective object complement"
    return parsed


# 목적보어(object complement)가 있는데, 앞쪽 목적어를 nsubj(subject)로 잘못 태깅하는 경우 예외처리
def repair_object_from_complement(parsed):
    for item in parsed:
        if item.get("role") in ["noun object complement", "adjective object complement"]:
            complement_children = item.get("children", [])
            
            # ✅ 동사 기준 가장 가까운 compound 중 object 후보 필터링
            compound_candidates = [
                t for t in parsed 
                if (
                    t["idx"] in complement_children and
                    t.get("dep") == "compound" and
                    t.get("pos") == "NOUN" and
                    t.get("head_idx") == item["idx"]
                )
            ]
            
            if compound_candidates:
                # 🔹 가장 낮은 idx (동사에 가까운 단어) 선택
                compound_candidates.sort(key=lambda x: x["idx"])
                compound_candidates[0]["role"] = "object"

            # 보완: 종종 주어를 object로 잘못 넣기도 함 (이건 그대로 유지)
            for t in parsed:
                if t.get("dep") == "nsubj" and t.get("idx") in complement_children:
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
                    "adjective subject complement",
                    "noun object complement",
                    "adjective object complement"  # 🔧 보어도 연결되게!
                ]:
                    combine.append({"text": t["text"], "role": r})
                    # ✅ 보완: indirect object가 자식 갖고 있으면 그 중 direct object도 연결
                    if r == "indirect object":
                        children = [c for c in all_tokens if c.get("head_idx") == t["idx"]]
                        for c in children:
                            if c.get("role") in ["direct object", "object"]:
                                combine.append({"text": c["text"], "role": c["role"]})

    # ✅ Indirect object → direct object (SVOO 구조)
    if role == "indirect object":
        for t in all_tokens:
            if (
                t.get("role") in ["direct object", "object"] and
                t.get("head_idx") == token.get("head_idx")
            ):
                combine.append({"text": t["text"], "role": "direct object"})

    # ✅ Object → object complement (SVOC 구조)
    if role == "object":
        for t in all_tokens:
            t_role = t.get("role") or ""
            if "object complement" in t_role:

                # 🔹 기본 연결 조건: 보어가 object 자식이거나 반대
                if (
                    t.get("head_idx") == idx or
                    idx == t.get("head_idx")
                ):
                    combine.append({"text": t["text"], "role": t["role"]})
                    continue

                # 🔹 추가 연결 조건: 보어가 object보다 뒤에 있고, head는 동일한 동사
                if (
                    t["idx"] > idx and
                    t.get("dep") in {"advcl", "oprd", "xcomp", "ccomp"} and
                    t.get("pos") == "ADJ" and
                    t.get("head_idx") == token.get("head_idx")
                ):
                    combine.append({"text": t["text"], "role": t["role"]})

    # ✅ Preposition → prepositional object
    if role == "preposition":
        for t in all_tokens:
            if t.get("head_idx") == idx and t.get("role") == "prepositional object":
                combine.append({"text": t["text"], "role": "prepositional object"})

    return combine if combine else None



def assign_level_triggers(parsed):
    """
    절 트리거(dep in trigger_deps)가 감지되면,
    절의 시작 위치에 해당하는 토큰에 level = 0.5 부여

    - relcl, advcl, ccomp, xcomp: children 중 가장 앞에 오는 토큰
    - acl: 트리거 단어 자신
    """
    trigger_deps = ["relcl", "acl", "advcl", "ccomp", "xcomp"]

    for token in parsed:
        if token["dep"] not in trigger_deps:
            continue

        if not is_valid_clause_trigger(token):
            continue

        dep = token["dep"]
        token_idx = token["idx"]

        if dep == "acl":
        # acl은 현재 단어가 절 시작임
            token["level"] = 0.5
            continue

        # children은 객체가 필요함 (idx 리스트 아님)
        children = [t for t in parsed if t["head_idx"] == token_idx and t["idx"] != token_idx]

        if children:
            # 절 내 가장 앞에 오는 토큰을 트리거로 판단
            first_token = min(children, key=lambda x: x["idx"])
            first_token["level"] = 0.5

    return parsed

def assign_level_ranges(parsed):
    """
    종속절을 담당하는 dep (relcl, acl, advcl, ccomp, xcomp)에 따라
    해당 절 범위에 level 값을 부여한다.
    
    - relcl, advcl, ccomp, xcomp: 해당 토큰 + children → 범위 계산
    - acl: 해당 토큰부터 children 포함하여 범위 계산 (자기자신이 연결어)
    
    그리고 마지막에 level=None인 토큰들에 대해 level=0을 부여한다.
    """

    clause_deps = ["relcl", "acl", "advcl", "ccomp", "xcomp"]

    current_level = 1  # 시작은 1부터 (0은 최상위 절용)

    for token in parsed:
        dep = token.get("dep")
        if dep not in clause_deps:
            continue
        
        if not is_valid_clause_trigger(token):
            continue

        token_idx = token["idx"]
        clause_tokens = [token]  # 시작은 자기 자신 포함

        # ✅ children도 절 범위에 포함
        children = [t for t in parsed if t["head_idx"] == token_idx]
        clause_tokens.extend(children)

        # ✅ 절 범위 시작 ~ 끝 계산
        start_idx = min(t["idx"] for t in clause_tokens)
        end_idx = max(t["idx"] for t in clause_tokens)

        # ✅ level 부여
        for t in parsed:
            if start_idx <= t["idx"] <= end_idx:
                t["level"] = current_level

        # ✅ 연결어에는 .5 추가
        if dep == "acl":
            token["level"] = current_level - 0.5  # 연결어는 바로 이전 절에서 이어짐
        else:
            # 연결어 후보: 절 범위 앞 단어 중 연결사 역할
            connector = min(clause_tokens, key=lambda x: x["idx"])
            connector["level"] = current_level - 0.5

        current_level += 1

    # ✅ 최상위 절 level=None → level=0 으로 설정
    for t in parsed:
        if t.get("level") is None:
            t["level"] = 0

    return parsed

# 목적보어를 계층 유발 요소로 태깅해 level이 발생하는 것에 대한 예외처리 함수임
def is_valid_clause_trigger(token: dict) -> bool:
    """
    절(clause) 트리거로 사용할 수 있는 토큰인지 판별합니다.

    예외 조건:
    - dep가 clause용 dep (ccomp, xcomp, advcl 등)가 아님
    - 보어 역할인 경우 (object complement 등)
    향후 조건이 더 생기면 여기에 추가합니다.
    """
    dep = token.get("dep")
    role = token.get("role")
    pos = token.get("pos") 

    clause_deps = ["ccomp", "xcomp", "advcl", "acl", "relcl"]

    if dep not in clause_deps:
        return False

    if role in ["adjective object complement", "noun object complement"]:
        return False

    # ✅ 예외: ADJ인데 dep=advcl 인 경우는 절 아님
    if dep == "advcl" and pos == "ADJ":
        return False
    
    # 향후 더 예외조건이 생기면 여기에 추가
    return True

def repair_level_within_prepositional_phrases(parsed):
    """
    prep의 목적어 pobj를 찾을 때:
    - dep == 'pobj'
    - head_idx == prep.idx
    - prep의 실제 children에 포함
    
    세 가지 모두 만족해야 함.
    
    level이 다르면 prep 기준으로 보정.
    """

    for prep in parsed:
        if prep.get("dep") != "prep":
            continue

        prep_level = prep.get("level")
        if prep_level is None:
            continue

        # ✅ prep의 children 목록 확보
        children = [t for t in parsed if t.get("head_idx") == prep["idx"]]
        child_ids = {t["idx"] for t in children}

        # ✅ 모든 토큰 중에서 pobj 후보 찾기 (이중 조건 적용)
        pobj_candidates = [
            t for t in parsed
            if t.get("dep") == "pobj"
            and t.get("head_idx") == prep["idx"]
            and t["idx"] in child_ids
        ]

        for pobj in pobj_candidates:
            pobj_level = pobj.get("level")

            # level이 같으면 보정 필요 없음
            if pobj_level == prep_level:
                continue

            # ✅ prep ~ pobj 범위 추출
            start = min(prep["idx"], pobj["idx"])
            end = max(prep["idx"], pobj["idx"])

            for t in parsed:
                if start <= t["idx"] <= end:
                    t["level"] = prep_level

    return parsed


# 처음 나오는 조동사와 본동사 사이를 .(점)으로 연결 시켜줌
def apply_modal_bridge_symbols_all_levels(parsed, sentence):
    modals = {"will", "would", "shall", "should", "can", "could", "may", "might", "must"}
    line_length = memory["sentence_length"]

    # 🔍 symbols_by_level에 이미 생성된 레벨만 순회
    for level, line in memory["symbols_by_level"].items():
        print(f"🧪 [bridge] processing existing level {level}")

        modal_token = next(
            (t for t in parsed if t["pos"] == "AUX" and t["dep"] in {"aux", "auxpass"} and t.get("level") == level),
            None
        )
        if not modal_token:
            continue

        main_verbs = [t for t in parsed if t.get("role") == "verb" and t.get("level") == level]
        if not main_verbs:
            continue
        verb_token = main_verbs[-1]

        modal_idx = modal_token["idx"]
        verb_idx = verb_token["idx"]
        start, end = sorted([modal_idx, verb_idx])

        # ✅ 의문문 판단
        has_subject_between = any(
            t.get("role") == "subject" and start < t["idx"] < end
            for t in parsed
        )

        if has_subject_between:
            if line[modal_idx] == " ":
                line[modal_idx] = "n" #∩
        else:
            if line[modal_idx] == " ":
                line[modal_idx] = "."

        for i in range(start + 1, end):
            if line[i] == " ":
                line[i] = "."


# 아무 심볼도 안 찍힌 줄이면 memory에서 아예 제거
def clean_empty_symbol_lines():
    """
    memory["symbols_by_level"] 중 내용이 전부 공백인 줄은 제거한다.
    """
    keys_to_remove = []
    for level, line in memory["symbols_by_level"].items():
        if all(c == " " for c in line):
            keys_to_remove.append(level)

    for level in keys_to_remove:
        del memory["symbols_by_level"][level]


def set_verb_attribute(parsed):
    """
    parsed 데이터를 기반으로 시제, 상, 태 정보를 추출해서 memory["verb_attribute"]에 저장.
    """
    verb_chain = [t for t in parsed if t["pos"] in {"AUX", "VERB"}]
    if not verb_chain:
        memory["verb_attribute"] = {}
        return

    first = verb_chain[0]
    last = verb_chain[-1]
    morph = last.get("morph", {})

    # ✅ 안전한 tense 파싱
    raw_tense = morph.get("Tense")
    if isinstance(raw_tense, str):
        tense = raw_tense.lower()
    elif isinstance(raw_tense, list) and raw_tense:
        tense = raw_tense[0].lower()
    else:
        tense = "pres"

    symbol_map = {}
    aspect = []
    voice = None

    # 시제
    symbol_map[first["idx"]] = ">" if tense == "past" else "|"

    # 완료
    for t in verb_chain:
        if t["lemma"] == "have":
            symbol_map[t["idx"]] = "P"
            aspect.append("perfect")

    # 진행
    for t in verb_chain:
        if t["lemma"] == "be" and t.get("tag") == "VBG":
            symbol_map[t["idx"]] = "i"
            aspect.append("progressive")

    # 수동
    for t in verb_chain:
        if t["lemma"] == "be" and t.get("tag") == "VBN":
            symbol_map[t["idx"]] = "^"
            voice = "passive"

    memory["verb_attribute"] = {
        "tense": tense,
        "aspect": aspect,
        "voice": voice,
        "main_verb": last["text"],
        "verb_chain": [t["text"] for t in verb_chain],
        "tense_position": first["idx"],
        "symbol_map": symbol_map
    }


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
            "tense": morph.get("Tense"), "aspect": morph.get("Aspect"), "voice": morph.get("Voice"), "form": morph.get("VerbForm"),
            "morph": morph, "lemma": token.lemma_, "is_stop": token.is_stop,
            "is_punct": token.is_punct, "is_alpha": token.is_alpha, "ent_type": token.ent_type_,
            "is_title": token.is_title, "children": [child.text for child in token.children]
        })

    # 규칙 기반 파싱
    parsed = rule_based_parse(tokens)

    # ✅ 보어 형용사 보정: ADJ인데 object로 된 경우
    parsed = assign_adj_object_complement_when_compound_object(parsed)

    # ✅ 보어 기준으로 object를 복원 (compound인 경우 등)
    parsed = repair_object_from_complement(parsed)

    # ✅ NEW: advcl+ADJ 보어 보정
    parsed = assign_adj_complement_for_advcl_adjective(parsed)

    # SVOO 관련 보정(indirect object role만 있는 경우)
    parsed = recover_direct_object_from_indirect(parsed)


    # ✅ 요기! 모든 보정 끝난 후에 combine 추론
    for t in parsed:
        combine = guess_combine(t, parsed)
        if combine:
            t["combine"] = combine

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

    # ✅ 절 분기 트리거 부여 (0.5 level)
    parsed = assign_level_triggers(parsed)

    print("\n📍 Level trigger check")
    for t in parsed:
        if "level" in t and isinstance(t["level"], float):
            print(f"→ TRIGGER: {t['text']} (level: {t['level']})")

    # level 분기 전파
    parsed = assign_level_ranges(parsed)

    # ✅ 📍 level 보정: prep-pobj 레벨 통일
    parsed = repair_level_within_prepositional_phrases(parsed)

    set_verb_attribute(parsed)

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

    # ⬇️ 여기서 combine 연결선을 _ 로 그려줌!
    for item in parsed:
        combine = item.get("combine")
        level = item.get("level")
        idx1 = item.get("idx")

        if not combine or level is None:
            continue

        for comb in combine:
            idx2 = next((t["idx"] for t in parsed if t["text"] == comb["text"]), None)
            if idx2 is None:
                continue

            # 같은 레벨 줄에 밑줄 채우기
            lvl = int(level)  # level이 float이면 int로 변환

            line = symbols_by_level.setdefault(lvl, [" " for _ in range(line_length)])
            start = min(idx1, idx2)
            end = max(idx1, idx2)

            for i in range(start + 1, end):
                if line[i] == " ":
                    line[i] = "_"


# ◎ memory["symbols"] 내용을 출력하기 위해 만든 함수
def symbols_to_diagram(sentence: str):
    output_lines = []

    line_length = memory["sentence_length"]
    parsed = memory.get("parsed")

    # ✅ 새 방식으로 시제/상/태 symbol map 출력
    tav_line = [" " for _ in range(line_length)]
    symbol_map = memory.get("verb_attribute", {}).get("symbol_map", {})
    for idx, symbol in symbol_map.items():
        if 0 <= idx < line_length:
            tav_line[idx] = symbol
    output_lines.append("".join(tav_line))  # ← 첫 줄로 출력

    # ✅ 문장 텍스트 줄
    output_lines.append(sentence)

    # ✅ 디버깅 로그 추가
    print("\n🧪 [DEBUG] symbols_by_level keys →", sorted(memory["symbols_by_level"].keys()))

    # ✅ bridge(∩) 및 ○□ 심볼 출력
    if parsed:
        apply_modal_bridge_symbols_all_levels(parsed, sentence)

#   clean_empty_symbol_lines()

    for level in sorted(memory["symbols_by_level"]):
        output_lines.append(''.join(memory["symbols_by_level"][level]))

    return '\n'.join(output_lines)


def t(sentence: str):
    print(f"\n📘 Sentence: {sentence}")

    # ✅ 메모리 먼저 초기화 (문장 길이 기반 설정 포함)
    init_memorys(sentence)

    # ✅ spaCy 파싱 + 역할 분석
    parsed = spacy_parsing_backgpt(sentence)
    memory["parsed"] = parsed

    # ✅ 동사덩어리 분석: 시제/상/태 출력
    verb_chain = [t for t in parsed if t["pos"] in {"AUX", "VERB"}]
    if verb_chain:
        first = verb_chain[0]
        last = verb_chain[-1]
        tense = last.get("morph", {}).get("Tense", ["Pres"])[0].lower()

        print(f"\n⏳ Verb Chain:")
        print("  → ", " → ".join(t['text'] for t in verb_chain))
        print(f"  Tense: {tense.upper()}  (based on main verb '{last['text']}')")
        print(f"  Tense mark position: idx({first['idx']})")

        for t in verb_chain:
            tag = t.get("tag")
            if t["lemma"] == "have":
                print(f"  ✔ Perfect (P) at idx({t['idx']}) [{t['text']}]")
            if t["lemma"] == "be" and tag == "VBG":
                print(f"  ✔ Progressive (i) at idx({t['idx']}) [{t['text']}]")
            if t["lemma"] == "be" and tag == "VBN":
                print(f"  ✔ Passive (^) at idx({t['idx']}) [{t['text']}]")

    # ⏳ Verb Symbol Map 확인 (선택적 디버깅)
    print("\n🧩 Verb Symbol Map:")
    for idx, symbol in memory.get("verb_attribute", {}).get("symbol_map", {}).items():
        print(f"  idx({idx}) → {symbol}")

    print("\n🔎 Combine Check:")
    for tok in parsed:
        if tok.get("combine"):
            print(f"  • {tok['text']} → {tok['combine']}")

    # ✅ morph 상세 출력
    print("\n📊 Full Token Info with Annotations:")
    doc = nlp(sentence)
    for token in doc:
        morph = token.morph.to_dict()
        idx = token.idx
        text = token.text
        role = next((t.get("role") for t in parsed if t["idx"] == idx), None)
        combine = next((t.get("combine") for t in parsed if t["idx"] == idx), None)
        level = next((t.get("level") for t in parsed if t["idx"] == idx), None)

        combine_str = (
            "[" + ", ".join(f"{c['text']}:{c['role']}" for c in combine) + "]"
            if combine else "None"
        )

        child_texts = [child.text for child in token.children]

        print(f"● idx({idx}), text({text}), role({role}), combine({combine_str}), level({level})")
        print(f"  POS({token.pos_}), DEP({token.dep_}), TAG({token.tag_}), HEAD({token.head.text})")
        print(f"  lemma({token.lemma_}), is_stop({token.is_stop}), is_punct({token.is_punct}), is_title({token.is_title})")
        print(f"  morph({morph})")
        print(f"  children({child_texts})")
        print("")

    # ✅ 도식화 및 출력
    apply_symbols(parsed)
    print("🛠 Diagram:")
    print(symbols_to_diagram(sentence))


# 묶음 테스트 함수
def t1(sentence: str):
    # ✅ 메모리 먼저 초기화 (문장 길이 기반 설정 포함)
    init_memorys(sentence)

    # ✅ spaCy 파싱 + 역할 분석
    parsed = spacy_parsing_backgpt(sentence)
    memory["parsed"] = parsed
    # ✅ 도식화 및 출력
    apply_symbols(parsed)
    print("🛠 Diagram:")
    print(symbols_to_diagram(sentence))



# ◎ 모듈 외부 사용을 위한 export
__all__ = [
    "rule_based_parse",
    "guess_role",
    "assign_noun_complement_for_SVOC_noun_only",
    "repair_object_from_complement",
    "guess_combine",
    "assign_level_triggers",
    "assign_level_ranges",
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
    memory["parsed"] = parsed
    apply_symbols(parsed)                              # parsed 결과에 따라 심볼들을 메모리에 저장장
    return {"sentence": request.sentence,
            "diagramming": symbols_to_diagram(request.sentence),
            "verb_attribute": memory.get("verb_attribute", {})
    }

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
            "tense": morph.get("Tense"), "aspect": morph.get("Aspect"), "form": morph.get("VerbForm"),
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
##
