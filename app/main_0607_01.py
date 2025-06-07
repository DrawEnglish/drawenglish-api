import os, json, re
import spacy
import uvicorn
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

# 환경 변수에서 모델명 가져오기, 없으면 'en_core_web_sm' 기본값
model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")

try:
    nlp = spacy.load(model_name)
except OSError:
    # 모델이 없으면 다운로드 후 다시 로드
    from spacy.cli import download
    download(model_name)
    nlp = spacy.load(model_name)

# ◎ 심볼 매핑
role_to_symbol = {
    "verb": "◯",
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

verb_attr_symbol = {
    "present tense": "|",
    "past tense": ">",
    "perfect aspect": "P",
    "progressive aspect": "i",
    "passive voice": "^",
    "subjunctive mood": "》"
}

verbals_symbol = {
    "R": "@",          # bare infinitive or root
    "to R": "to@",     # to infinitive
    "R-ing": "@ing",   # gerund or present participle
    "R-ed": "@ed"      # past participle
}

relative_words_symbol = {
    "relative pronoun": "[X]",
    "relative adjective": "(_)",
    "relative adverb": "<_>",
    "compound relative pronoun": "[e]",
    "compound relative adjective": "(e)",
    "compound relative adverb": "<e>",
    "interrogative pronoun": "[?]",
    "interrogative adjective": "(?)",
    "interrogative adverb": "<?>"
}

# 전체 심볼 통합 딕셔너리
symbols = {
    "role": role_to_symbol,
    "verb_attr": verb_attr_symbol,
    "verbals": verbals_symbol,
    "relatives": relative_words_symbol
}

# ◎ 메모리 구조 / 메모리 초기화화
memory = {
    "symbols_by_level": {},
    "symbols": symbols
}


# level 발생 트리거 dep 목록 (전역으로 통일)
level_trigger_deps = [
    "relcl", "acl", "advcl", "advmodcl", "ccomp", "xcomp", "csubj", "parataxis"
]

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

modalVerbs_present = {"will", "shall", "can", "may", "must"}
modalVerbs_past = {"would", "should", "could", "might"}
modalVerbs_all = modalVerbs_present | modalVerbs_past

beVerbs = {"be", "am", "are", "is", "was", "were", "been", "being"}

notbeLinkingVerbs_onlySVC = {
    "become", "come", "go", "fall", "sound", "look", "smell", "taste", "seem"
}
notbeLinkingVerbs_SVCSVO = {
    "get", "turn", "grow", "feel"
}
netbeLinkingVerbs_all = notbeLinkingVerbs_SVCSVO | notbeLinkingVerbs_onlySVC

dativeVerbs = {
    "give", "send", "offer", "show", "lend", "teach", "tell", "write", "read", "promise",
    "sell", "pay", "pass", "bring", "buy", "ask", "award", "grant", "feed", "hand", "leave", "save", 
    "bring", "bake", "build", "cook", "sing", "make"  # dative verb로 사용 드문 것들
}


# spaCy가 전치사로 오인 태깅하는 특수 단어들
blacklist_preposition_words = {"due", "according"}


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
        t["role1"] = None
        t["role2"] = None

        # role 추론
        role1 = guess_role(t, tokens)
        if role1:
            t["role1"] = role1  # combine에서 쓰일 수 있음

    # 'name'과 같은 동사가 있는 SVOC구조에서 목적보어를 잘못 태깅하는 것 보정 함수
    result = tokens  # 기존 tokens을 수정하며 계속 사용
    result = assign_noun_complement_for_SVOC_noun_only(result)

    # ✅ 보어 기반 object 복구 자동 적용
    result = repair_object_from_complement(result)        

    return result


# role 추론 함수
def guess_role(t, all_tokens=None):  # all_tokens 추가 필요
    dep = t.get("dep")
    pos = t["pos"]
    head_idx = t.get("head_idx")

    # ✅ Subject
    if dep in ["nsubj", "nsubjpass"]:
        return "subject"

    # ✅ Main Verb: be동사 포함, 종속절도 고려
    if pos in ["VERB", "AUX"] and (dep in level_trigger_deps or dep == "root"):
        return "verb"

    # ✅ 등위접속사 다음 병렬 동사 (conj)도 verb role 부여
    if pos == "VERB" and dep == "conj":
        if any(t["idx"] == head_idx and t.get("role1") == "verb" for t in all_tokens):
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


    # ✅ Preposition: 기본 prep, agent + 보완 (pcomp), 단 blacklist 단어는 제외
    if (
        (dep in ["prep", "agent"] or (dep == "pcomp" and pos == "ADP" and t.get("tag") == "IN"))
        and t["text"].lower() not in blacklist_preposition_words
    ):
        return "preposition"

    # ✅ Prepositional Object (by ~pobj 구조 커버)
    # 위 Preposition Role결정시 blacklist 단어에 의해 전치사의 목적어를 못찾는 문제 보정
    if dep == "pobj":
        head_token = next((t for t in all_tokens if t["idx"] == head_idx), None)
        if head_token and (
            head_token.get("role1") == "preposition"
            or (
                head_token["text"].lower() in blacklist_preposition_words and
                (
                    head_token.get("pos") == "ADP" or
                    head_token.get("dep") == "prep" or
                    head_token.get("tag") == "IN"
                )
            )
        ):
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
        if token.get("role1") == "indirect object":
            token_idx = token.get("idx")

            for child in parsed:
                if (
                    child.get("head_idx") == token_idx and
                    child.get("dep") == "appos" and
                    child.get("pos") in {"NOUN", "PROPN"}
                ):
                    child["role1"] = "direct object"
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
                    t["role1"] = "noun object complement"
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
                    t["role1"] = "adjective object complement"

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
            (t for t in parsed if t.get("head_idx") == verb_idx and t.get("role1") in ["object", "direct object"]),
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
                t["role1"] = "adjective object complement"
    return parsed


# 목적보어(object complement)가 있는데, 앞쪽 목적어를 nsubj(subject)로 잘못 태깅하는 경우 예외처리
def repair_object_from_complement(parsed):
    for item in parsed:
        if item.get("role1") in ["noun object complement", "adjective object complement"]:
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
                compound_candidates[0]["role1"] = "object"

            # 보완: 종종 주어를 object로 잘못 넣기도 함 (이건 그대로 유지)
            for t in parsed:
                if t.get("dep") == "nsubj" and t.get("idx") in complement_children:
                    t["role1"] = "object"

    return parsed


# combine 추론 함수
def guess_combine(token, all_tokens):
    role1 = token.get("role1")
    idx = token.get("idx")
    combine = []

    # ✅ Verb → object / complement (SVO, SVC)
    if role1 == "verb":
        for t in all_tokens:
            if (
                t.get("head_idx") == idx
                and t["idx"] > idx  # 🔧 오른쪽 방향 연결만 허용
            ):
                r = t.get("role1")
                if r in [
                    "object",
                    "indirect object",
                    "noun subject complement",
                    "adjective subject complement",
                    "noun object complement",
                    "adjective object complement"  # 🔧 보어도 연결되게!
                ]:
                    combine.append({"text": t["text"], "role1": r, "idx": t["idx"]})
                    # ✅ 보완: indirect object가 자식 갖고 있으면 그 중 direct object도 연결
                    if r == "indirect object":
                        children = [c for c in all_tokens if c.get("head_idx") == t["idx"]]
                        for c in children:
                            if (
                                c.get("role1") in ["direct object", "object"]
                                and c["idx"] > t["idx"]  # 🔧 핵심 추가
                            ):
                                combine.append({"text": c["text"], "role1": c["role1"], "idx": c["idx"]})

    # ✅ Indirect object → direct object (SVOO 구조)
    if role1 == "indirect object":
        for t in all_tokens:
            if (
                t.get("role1") in ["direct object"] and
                t.get("head_idx") == token.get("head_idx")
                and t["idx"] > token["idx"]  # 🔧 오른쪽 방향만 연결
            ):
                combine.append({"text": t["text"], "role1": "direct object", "idx": t["idx"]})

    # ✅ Object → object complement (SVOC 구조)
    if role1 == "object":
        for t in all_tokens:
            t_role = t.get("role1") or ""
            if "object complement" in t_role:
                if (
                    t.get("head_idx") == idx or
                    idx == t.get("head_idx")
                ):
                    combine.append({"text": t["text"], "role1": t["role1"], "idx": t["idx"]})
                    continue

                # 🔹 추가 연결 조건: 보어가 object보다 뒤에 있고, head는 동일한 동사
                if (
                    t["idx"] > idx and
                    t.get("dep") in {"advcl", "oprd", "xcomp", "ccomp"} and
                    t.get("pos") == "ADJ" and
                    t.get("head_idx") == token.get("head_idx")
                ):
                    combine.append({"text": t["text"], "role1": t["role1"], "idx": t["idx"]})

    # ✅ Preposition → prepositional object
    if role1 == "preposition":
        for t in all_tokens:
            if t.get("head_idx") == idx and t.get("role1") == "prepositional object":
                combine.append({"text": t["text"], "role1": "prepositional object", "idx": t["idx"]})

        # 2️⃣ 예외 보정: head가 due/according인데, 이 token이 그 뒤의 "to"일 경우
    for t in all_tokens:
        if (
            t.get("role1") == "prepositional object" and
            t.get("head_idx") in [
                b["idx"] for b in all_tokens if b["text"].lower() in blacklist_preposition_words
            ]
        ):
            # 👉 head token
            head_idx = t.get("head_idx")
            head_token = next((tok for tok in all_tokens if tok["idx"] == head_idx), None)

            # ✅ head_token 다음에서 "to" 찾기
            to_token = next(
                (
                    tok for tok in all_tokens
                    if tok["text"].lower() == "to" and tok["idx"] > head_token["idx"]
                ),
                None
            )

            # ✅ 이 token이 그 "to"일 경우만 연결
            if to_token and to_token["idx"] == idx:
                combine.append({"text": t["text"], "role1": t["role1"], "idx": t["idx"]})

    # ✅ combine 있을 경우만 반환
    return combine if combine else None


def assign_level_triggers(parsed):
    """
    절 트리거(dep in trigger_deps)가 감지되면,
    절의 시작 위치에 해당하는 토큰에 level = 0.5 부여

    - relcl, advcl, ccomp, xcomp: children 중 가장 앞에 오는 토큰
    - acl: 트리거 단어 자신
    """

    for token in parsed:
        if token["dep"] not in level_trigger_deps:
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

def is_nounchunk_trigger(token):

    # 명사절 첫단어 트리거 조건 : SCONJ + mark + IN
    if token.get("pos") == "SCONJ" and token.get("dep") == "mark" and token.get("tag") == "IN":
        return True

    # to 부정사
    if (
        token.get("pos") == "PART" and token.get("dep") == "aux" and token.get("tag") == "TO" and
        token.get("lemma", "").lower() == "to"
    ):
        return True

    return False

def is_adverbchunk_trigger(token):

    # 부사절 첫단어 트리거 조건 : SCONJ + mark/advmod + IN/WRB
    return (
        token.get("pos") == "SCONJ" and
        token.get("dep") in {"mark", "advmod"} and
        token.get("tag") in {"IN", "WRB"}
    )


def assign_chunk_role2(parsed):

    # 명사덩어리/부사덩어리 첫단어 role2에 해당값 부여 

    chunk_info_list = []

    # 계층발생요소(level x.5단어)만 아래 소스 처리
    for token in parsed:
        level = token.get("level")
        if not (isinstance(level, float) and level % 1 == 0.5):
            continue

        #계층발생요소의 헤드 값이 있으면 아래 소스 처리
        head_idx = token.get("head_idx")
        head_token = next((t for t in parsed if t["idx"] == head_idx), None)
        if not head_token:
            continue

        head_dep = head_token.get("dep")

        # 명사덩어리 판단 : 계층발생요소 헤드의 dep가(ccomp, xcomp) 이고, 
        # 계층발생요소가 is_nounchunk_trigger에 걸리면,
        if head_dep in {"ccomp", "xcomp"} and is_nounchunk_trigger(token):

            # 계층발생요소의 head의 head 찾기 head값이 있으면 아래 소스 처리
            head2_idx = head_token.get("head_idx")
            head2_token = next((t for t in parsed if t["idx"] == head2_idx), None)
            if not head2_token:
                continue

            # 계층발생요소 head의 head인 상위동사(head2) lemma값 저장
            head2_lemma = head2_token.get("lemma", "")

            # 1) 명사덩어리 확정후 상위동사가 be동사와 LinkingVerbs이면 보어 확정
            # 명사덩어리 첫단어의 role2에 'noun subject complement'(명사주어보어)값 저장
            if head2_lemma in beVerbs or head2_lemma in notbeLinkingVerbs_onlySVC:
                token["role2"] = "noun subject complement"

            # 2) 상위동사가 dativeVerbs일때 상위동사level 단어들의 role1에 objedct, indirect object가 있으면
            # 명사덩어리 첫단어의 role2에 'direct object'(직접목적어)값 저장
            # 아니면 role2에 'object'(목적어)값 저장
            elif head2_lemma in dativeVerbs:
                current_level = int(token.get("level", 0))  # 0.5 -> 0
                # 현재 레벨의 토큰들
                level_tokens = [t for t in parsed if int(t.get("level", -1)) == current_level]
                has_obj_or_iobj = any(
                    t.get("role1") in {"object", "indirect object"} for t in level_tokens
                )
                if has_obj_or_iobj:
                    token["role2"] = "direct object"
                else:
                    token["role2"] = "object"

            # 앞 모든 조건에 안걸리면 명사덩어리 첫단어의 rele2에 'object'(목적어) 값 저장
            else:
                token["role2"] = "object"


        # 주어 명사덩어리 확정 : 덩어리요소 첫단어의 head의 dep가 csubj, nsubj, nsubjpass이고,
        # is_nounchunk_trigger() 함수에 걸리면 role2에 'chunk_subject'값 입력
        if head_dep in {"csubj", "nsubj", "nsubjpass"} and is_nounchunk_trigger(token):
            token["role2"] = "chunk_subject"

        # 부사덩어리 확정 : 덩어리요소 첫단어의 head의 dep가 advcl이고,
        # is_adverbchunk_trigger() 함수에 걸리면 role2에 'chunk_adverb_modifier'값 입력
        if head_dep == "advcl" and is_adverbchunk_trigger(token):
            token["role2"] = "chunk_adverb_modifier"

        
        # ✅ 덩어리 정보 수집 (끝 토큰 찾기 + 시작 토큰 info)
        children_tokens = [child for child in parsed if child.get("head_idx") == head_idx]
        children_tokens.append(head_token)
        if not children_tokens:
            continue

        children_tokens.sort(key=lambda x: x["idx"])
        end_token = children_tokens[-1]

        # 끝 토큰이 구두점(. ! ?)이면 그 앞 토큰 사용
        if (
            end_token.get("pos") == "PUNCT" and
            end_token.get("text") in {".", "!", "?"} and
            len(children_tokens) >= 2
        ):
            end_token = children_tokens[-2]

        end_idx = end_token.get("idx")
        end_text = end_token.get("text", "")
        end_idx_adjusted = end_idx + len(end_text) - 1

        first_level = int(token.get("level"))
        first_idx = token.get("idx")

        # 덩어리 유형별 role2 심볼 결정
        role2_to_symbol = {
            "object": "□",
            "direct object": "□",
            "noun subject complement": "[",
            # 🔥 앞으로 추가 가능:
            # "adjective subject complement": "(",
            # "chunk_subject": "[",
            # "chunk_adverb_modifier": "<",
        }

        role2 = token.get("role2")
        symbol = role2_to_symbol.get(role2)

        if symbol:
            chunk_info = {
                "first_idx": first_idx,
                "first_level": first_level,
                "symbol": symbol,
                "end_idx_adjusted": end_idx_adjusted,
            }
            chunk_info_list.append(chunk_info)

    return chunk_info_list


def apply_chunk_symbols_overwrite(chunk_info_list):
    """
    수집된 덩어리 정보 리스트를 바탕으로
    1) 덩어리 끝단어에 ] 심볼
    2) 덩어리 첫단어에 role2 심볼(□, [ 등) 찍기
    """
    symbols_by_level = memory["symbols_by_level"]
    line_length = memory["sentence_length"]

    for chunk in chunk_info_list:
        first_idx = chunk["first_idx"]
        first_level = chunk["first_level"]
        symbol = chunk["symbol"]
        end_idx_adjusted = chunk["end_idx_adjusted"]

        line = symbols_by_level.setdefault(first_level, [" " for _ in range(line_length)])

        # 1) 첫단어에 role2 심볼 찍기
        if 0 <= first_idx < len(line):
            line[first_idx] = symbol

        # 2) 끝단어 끝글자에 ] 심볼 찍기
        if 0 <= end_idx_adjusted < len(line):
            line[end_idx_adjusted] = "]"


def apply_chunk_function_symbol(parsed):
    """
    role2=chunk_subject인 토큰을 기준으로
    해당 절(start_idx ~ end_idx) 범위에 [ ] 심볼 부여
    """
    line_length = memory["sentence_length"]
    symbols_by_level = memory["symbols_by_level"]

    for token in parsed:
        role2 = token.get("role2")
        if not role2:
            continue

        level = token.get("level")
        if level is None:
            continue

        line = symbols_by_level.setdefault(int(level), [" " for _ in range(line_length)])

        start_idx = token["idx"]
        head_idx = token.get("head_idx")
        head_token = next((t for t in parsed if t["idx"] == head_idx), None)

        if not head_token:
            continue

        children_tokens = [child for child in parsed if child.get("head_idx") == head_idx]
        children_tokens.append(head_token)
        if not children_tokens:
            continue

        children_tokens.sort(key=lambda x: x["idx"])

        end_token = children_tokens[-1]

        if end_token.get("pos") == "PUNCT" and len(children_tokens) >= 2:
            end_token = children_tokens[-2]

        end_idx = end_token["idx"]
        end_idx_adjusted = end_idx + len(end_token["text"]) - 1

        # ✅ role2에 따라 심볼 다르게
        if role2 == "chunk_subject":
            left, right = "[", "]"
        elif role2 == "chunk_adverb_modifier":
            left, right = "<", ">"
        else:
            continue

        if 0 <= start_idx < line_length:
            line[start_idx] = left
        if 0 <= end_idx_adjusted < line_length:
            line[end_idx_adjusted] = right


def NounChunk_combine_apply_to_upverb(parsed):
    """
    명사덩어리 첫단어 role2가 object / direct object / noun subject complement일때
    상위 동사의 comnbin에 role2를 입력해주는 함수
    """
    for token in parsed:
        role2 = token.get("role2")
        # 명사덩어리 첫단어의 role2가 이 3개일때만 아래 소스 처리
        if role2 not in {"object", "direct object", "noun subject complement"}:
            continue

        # 명사덩어리 첫단어의 head(보통 동사)의 dep가 ccomp(종속접속사)일때만 아래 소스 처리
        head_idx = token.get("head_idx")
        head_token = next((t for t in parsed if t["idx"] == head_idx), None)
        if not head_token or head_token.get("dep") not in {"ccomp", "xcomp"}:
            continue

        # 명사덩어리 첫단어의 head의 head(상위 동사 head2)가 있으면 아래 소스 처리리
        head2_idx = head_token.get("head_idx")
        head2_token = next((t for t in parsed if t["idx"] == head2_idx), None)
        if not head2_token:
            continue
        if "combine" not in head2_token or not head2_token["combine"]:
            head2_token["combine"] = []

        # 🔥 상위 동사의 combine에 위 role2 3개중 1개(text, role2값, idx값) 입력
        head2_token["combine"].append({
            "text": token["text"],
            "role2": role2,
            "idx": token["idx"]
        })


def assign_level_ranges(parsed):
    """
    종속절을 담당하는 dep (relcl, acl, advcl, ccomp, xcomp)에 따라
    해당 절 범위에 level 값을 부여한다.
    
    - relcl, advcl, ccomp, xcomp: 해당 토큰 + children → 범위 계산
    - acl: 해당 토큰부터 children 포함하여 범위 계산 (자기자신이 연결어)
    
    그리고 마지막에 level=None인 토큰들에 대해 level=0을 부여한다.
    """

    current_level = 1  # 시작은 1부터 (0은 최상위 절용)

    for token in parsed:
        dep = token.get("dep")
        if dep not in level_trigger_deps:
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
    role1 = token.get("role1")
    pos = token.get("pos") 

    if dep not in level_trigger_deps:
        return False

    if role1 in ["adjective object complement", "noun object complement"]:
        return False

    # ✅ 예외: ADJ인데 dep=advcl 인 경우는 절 아님
    if dep == "advcl" and pos == "ADJ":
        return False
    
    # 향후 더 예외조건이 생기면 여기에 추가
    return True

def repair_level_within_prepositional_phrases(parsed):
    """
    전치사(prep 또는 agent)의 목적어(pobj) 레벨이 다를 경우
    전치사의 level 기준으로 범위 내 토큰들을 보정.
    """

    for prep in parsed:
        if prep.get("dep") not in {"prep", "agent"}:
            continue

        prep_level = prep.get("level")
        if prep_level is None:
            continue

        prep_idx = prep["idx"]

        # ✅ 모든 토큰 중에서 pobj 후보 찾기 (children 조건 제외)
        pobj_candidates = [
            t for t in parsed
            if t.get("dep") == "pobj" and t.get("head_idx") == prep_idx
        ]

        for pobj in pobj_candidates:
            pobj_level = pobj.get("level")

            if pobj_level == prep_level:
                continue  # 이미 동일하면 건너뜀

            # ✅ prep ~ pobj 사이 범위를 찾아 level 보정
            start = min(prep_idx, pobj["idx"])
            end = max(prep_idx, pobj["idx"])

            for t in parsed:
                if start <= t["idx"] <= end:
                    t["level"] = prep_level
                    t["level_corrected_from_prep"] = True  # 디버깅용 표시

    return parsed


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


# 동사덩어리(verb chain) 하나 받아서 시제/상/태 분석하고 symbol_map 반환하는 함수.
def set_verbchunk_attributes(chain):

    symbol_map = {}
    aspect = []
    voice = None

    if not chain:
        return symbol_map, aspect, voice

    verb_attr = memory["symbols"]["verb_attr"]

    # 맨 앞 토큰
    first = chain[0]
    first_lemma = first.get("lemma", "").lower()
    first_pos = first.get("pos")
    first_dep = first.get("dep")
    first_tag = first.get("tag")

    # ✅ P1. 맨앞 modal 여부
    if first_pos == "AUX" and first_dep == "aux" and first_tag == "MD":
        if first_lemma in modalVerbs_present:
            symbol_map[first["idx"]] = verb_attr["present tense"]
        elif first_lemma in modalVerbs_past:
            symbol_map[first["idx"]] = verb_attr["past tense"]

    # ✅ P2. 중간 조동사들 처리
    for t in chain:
        pos = t.get("pos")
        dep = t.get("dep")
        tag = t.get("tag")
        text = t.get("text", "").lower()

        # aux, auxpass만 조동사
        if not (pos == "AUX" and dep in {"aux", "auxpass"}):
            break  # 조동사 아니면 (즉 본동사) -> P3로

        # 조동사 시제 (fin)
        verbform = t.get("morph", {}).get("VerbForm", "")
        tense = t.get("morph", {}).get("Tense", "")

        if verbform == "Fin":
            if tag in {"VBP", "VBZ"} or tense == "Pres":
                symbol_map[t["idx"]] = verb_attr["present tense"]
            elif tag == "VBD" or tense == "Past":
                symbol_map[t["idx"]] = verb_attr["past tense"]

        # 완료, 진행
        if text == "been" and tag == "VBN":
            symbol_map[t["idx"]] = verb_attr["perfect aspect"]
            if "perfect" not in aspect:
                aspect.append("perfect")
        elif text == "being" and tag == "VBG":
            symbol_map[t["idx"]] = verb_attr["progressive aspect"]
            if "progressive" not in aspect:
                aspect.append("progressive")

        # 원형 조동사(VB, Inf)는 아무것도 안찍고 continue
        if tag == "VB" and verbform == "Inf":
            continue

    # ✅ P3. 본동사 처리
    last = chain[-1]
    pos = last.get("pos")
    dep = last.get("dep")
    tag = last.get("tag")
    lemma = last.get("lemma", "").lower()

    if dep in level_trigger_deps or dep == "root":
        verbform = last.get("morph", {}).get("VerbForm", "")
        tense = last.get("morph", {}).get("Tense", "")

        if verbform == "Fin":
            if tag in {"VBP", "VBZ"} or tense == "Pres":
                symbol_map[last["idx"]] = verb_attr["present tense"]
            elif tag == "VBD" or tense == "Past":
                symbol_map[last["idx"]] = verb_attr["past tense"]


        # 완료/수동/진행
        if tag == "VBN":
            # 왼쪽으로 AUX aux/auxpass 찾아야 해
            for prev in reversed(chain[:-1]):
                if prev.get("pos") != "AUX":
                    continue
                prev_dep = prev.get("dep")
                prev_lemma = prev.get("lemma", "").lower()
                if prev_dep == "aux" and prev_lemma == "have":
                    symbol_map[last["idx"]] = verb_attr["perfect aspect"]
                    if "perfect" not in aspect:
                        aspect.append("perfect")
                    break
                elif prev_dep == "auxpass" and prev_lemma == "be":
                    symbol_map[last["idx"]] = verb_attr["passive voice"]
                    voice = "passive"
                    break

        elif tag == "VBG":
            symbol_map[last["idx"]] = verb_attr["progressive aspect"]
            if "progressive" not in aspect:
                aspect.append("progressive")

    return symbol_map, aspect, voice

# 문장의 전체 parsed 결과를 받아 동사덩어리별 시제/상/태 분석.
def set_allverbchunk_attributes(parsed):
    memory["verb_attribute_by_chain"] = []
    memory["verb_attribute"] = {}
    sentence_len = memory["sentence_length"]

    chains = []
    current_chain = []
    last_level = None

    # 동사덩어리 분리
    for token in parsed:
        level = token.get("level", 0)

        if last_level is None:
            last_level = level

        # 등위접속사가 나오면 동사덩어리 끊음 (종속접속사는 level 발생 부분에서 처리 가능)
        if (
            token.get("dep") in {"cc"} and token.get("pos") in {"CCONJ", "CONJ"}
        ):
            if current_chain:
                chains.append(current_chain)
                current_chain = []
            last_level = level

        # level 바뀔 때 끊기
        if last_level is not None and level != last_level:
            if current_chain:
                chains.append(current_chain)
                current_chain = []
            last_level = level

        # ✅ AUX, VERB 추가
        if token["pos"] in {"AUX", "VERB"}:
            current_chain.append(token)

    if current_chain:
        chains.append(current_chain)

    all_symbol_maps = {}

    # 각 chain 분석
    for chain in chains:
        if not chain:
            continue
        
        first = chain[0]
        last = chain[-1]

        symbol_map, aspect, voice = set_verbchunk_attributes(chain)

        # 저장 (디버깅용, 확장용)
        memory["verb_attribute_by_chain"].append({
            "aspect": aspect,
            "voice": voice,
            "main_verb": chain[-1]["text"],
            "verb_chain": [t["text"] for t in chain],
            "symbol_map": symbol_map
        })

        all_symbol_maps.update(symbol_map)

    memory["verb_attribute"] = {
        "symbol_map": all_symbol_maps,
        "main_verb": last["text"],
        "aspect": aspect,
        "voice": voice
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
            "dep": token.dep_.lower(), "head": token.head.text, "head_idx": token.head.idx,
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

    # level 분기 전파
    parsed = assign_level_ranges(parsed)

    # ✅ 📍 level 보정: prep-pobj 레벨 통일
    parsed = repair_level_within_prepositional_phrases(parsed)

    set_allverbchunk_attributes(parsed)

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
Each item must have: idx, text, role1, role2, and optionally combine/level.

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
        role1 = str(item.get("role1", "") or "").lower()
        level = item.get("level")

        if idx < 0 or level is None:
            continue

        # ✅ 0.5처럼 경계 레벨은 두 줄에 심볼 찍기
        levels = [level]
        if isinstance(level, float) and level % 1 == 0.5:
            levels = [int(level), int(level) + 1]

        # 
        symbol = role_to_symbol.get(role1)

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
            idx2 = comb.get("idx")  # ✅ text 비교 대신 idx 직접 사용
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


# 처음 나오는 조동사와 본동사 사이를 .(점)으로 연결 시켜줌, 레벨 순회하며(다른 레벨간 연결할일 없음), 기존 도형 있으면 안찍음
def apply_aux_to_mverb_bridge_symbols_each_levels(parsed, sentence):

    for modal_token in [t for t in parsed if t["pos"] == "AUX" and t["dep"] in {"aux", "auxpass"}]:
        level = modal_token.get("level")
        if level is None:
            continue

        line = memory["symbols_by_level"].get(level)
        if not line:
            continue

        modal_idx = modal_token["idx"]

        # ✅ 조동사 이후에 나오는 첫 번째 본동사(verb role)
        verb_token = next(
            (t for t in parsed
             if t.get("role1") == "verb"
             and t.get("level") == level
             and t["idx"] > modal_idx),
            None
        )
        if not verb_token:
            continue

        verb_idx = verb_token["idx"]
        start, end = sorted([modal_idx, verb_idx])

        # ✅ 의문문 판단
        has_subject_between = any(
            t.get("role1") == "subject" and start < t["idx"] < end
            for t in parsed
        )

        if has_subject_between:
            if line[modal_idx] == " ":
                line[modal_idx] = "∩"
        else:
            if line[modal_idx] == " ":
                line[modal_idx] = "."

        for i in range(start + 1, end):
            if line[i] == " ":
                line[i] = "."


# 동일레벨, 같은 절에 동사가 여러개 병렬 나열된 경우 동사덩어리 처음 요소와 끝요소를 .(점)으로 채워줌
def draw_dot_bridge_across_verb_group(parsed):
    line_length = memory["sentence_length"]
    symbols_by_level = memory["symbols_by_level"]
    visited = set()

    for token in parsed:
        # ✅ role 없이도 동사면 점선 연결 대상!
        if token.get("pos") not in {"AUX", "VERB"}:
            continue

        dep = token.get("dep", "").lower()
        if dep not in {"root", "conj", "xcomp", "ccomp"}:
            continue

        level = token.get("level")
        if level is None:
            continue

        idx1 = token["idx"]
        idx2 = None

        for t in parsed:
            if (
                t["idx"] > idx1 and
                t.get("level") == level and
                t.get("pos") in {"VERB", "AUX"} and
                t.get("dep", "").lower() in {"root", "conj", "xcomp", "ccomp"}
            ):
                has_subject_between = any(
                    s.get("role1") == "subject" and
                    s.get("level") == level and
                    idx1 < s["idx"] < t["idx"]
                    for s in parsed
                )
                if has_subject_between:
                    break
                idx2 = t["idx"]

        if idx2 and (idx1, idx2) not in visited:
            visited.add((idx1, idx2))
            line = symbols_by_level.setdefault(level, [" " for _ in range(line_length)])
            for i in range(idx1 + 1, idx2):
                if line[i] == " ":
                    line[i] = "."


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

    # ✅ bridge(∩) 및 ○□ 심볼 출력
    if parsed:
        apply_aux_to_mverb_bridge_symbols_each_levels(parsed, sentence)

#   clean_empty_symbol_lines()

    for level in sorted(memory["symbols_by_level"]):
        output_lines.append(''.join(memory["symbols_by_level"][level]))

    draw_dot_bridge_across_verb_group(parsed)

    return '\n'.join(output_lines)


def t(sentence: str):
    print(f"\n📘 Sentence: {sentence}")

    # ✅ 메모리 먼저 초기화 (문장 길이 기반 설정 포함)
    init_memorys(sentence)

    # ✅ spaCy 파싱 + 역할 분석
    parsed = spacy_parsing_backgpt(sentence)
    memory["parsed"] = parsed

    chunk_info_list = assign_chunk_role2(parsed)
    NounChunk_combine_apply_to_upverb(parsed)
    apply_symbols(parsed)
    apply_chunk_function_symbol(parsed)
    apply_chunk_symbols_overwrite(chunk_info_list)
    draw_dot_bridge_across_verb_group(parsed)

    # ✅ morph 상세 출력
    print("\n📊 Full Token Info with Annotations:")
    doc = nlp(sentence)
    for token in doc:
        morph = token.morph.to_dict()
        idx = token.idx
        text = token.text
        role1 = next((t.get("role1") for t in parsed if t["idx"] == idx), None)
        role2 = next((t.get("role2") for t in parsed if t["idx"] == idx), None)
        combine = next((t.get("combine") for t in parsed if t["idx"] == idx), None)
        level = next((t.get("level") for t in parsed if t["idx"] == idx), None)

        combine_str = (
            "[" + ", ".join(
                f"{c['text']}({c['idx']}):" + (
                    f"{c['role1']}" if 'role1' in c else ""
                ) + (
                    f"/{c['role2']}" if 'role2' in c else ""
                )
                for c in combine
            ) + "]"
            if combine else "None"
        )

        child_texts = [child.text for child in token.children]

        print(f"● idx({idx}), text({text}), role1({role1}), role2({role2}), combine({combine_str})")
        print(f"  level({level}), POS({token.pos_}), DEP({token.dep_}), TAG({token.tag_}), HEAD({token.head.text})")
        print(f"  lemma({token.lemma_}), is_stop({token.is_stop}), is_punct({token.is_punct}), is_title({token.is_title})")
        print(f"  morph({morph})")
        print(f"  children({child_texts})")
        print("")
        
    # ✅ 도식 출력
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
    chunk_info_list = assign_chunk_role2(parsed)
    NounChunk_combine_apply_to_upverb(parsed)
    apply_chunk_function_symbol(parsed)
    apply_symbols(parsed)
    apply_chunk_symbols_overwrite(chunk_info_list)
    draw_dot_bridge_across_verb_group(parsed)
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
    chunk_info_list = assign_chunk_role2(parsed)
    NounChunk_combine_apply_to_upverb(parsed)
    apply_chunk_function_symbol(parsed)
    apply_symbols(parsed)                              # parsed 결과에 따라 심볼들을 메모리에 저장장
    apply_chunk_symbols_overwrite(chunk_info_list)
    draw_dot_bridge_across_verb_group(parsed)
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
