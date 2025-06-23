import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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
model_name = os.getenv("SPACY_MODEL", "en_core_web_trf")

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

#verbals_symbol = {
#    "bare infinitive": "R",
#    "to infinitive": "to.R",
#    "gerund": "R.ing",
#    "present participle": "R.ing",
#    "past participele": "R.ed"
#}

relative_words_symbol = {
    "relative pronoun": "[X]",
    "relative adjective": "(X)",
    "relative adverb": "<X>",
    "compound relative pronoun": "[e]",
    "compound relative adjective": "(e)",
    "compound relative adverb": "<e>",
    "interrogative pronoun": "[?]",
    "interrogative adjective": "(?)",
    "interrogative adverb": "<?>"
}

# 전체 심볼 통합 딕셔너리
symbols_all = {
    "role": role_to_symbol,
    "verb_attr": verb_attr_symbol,
#    "verbals": verbals_symbol,
    "relatives": relative_words_symbol
}

# ◎ 메모리 구조 / 메모리 초기화화
memory = {
    "symbols_by_level": {},
    "symbols_all": symbols_all
}


# level 발생 트리거 dep 목록 (전역으로 통일)
level_trigger_deps = [
    "relcl", "acl", "advcl", "advmodcl", "ccomp", "xcomp", "csubj", "parataxis"
]

# guess_role() 함수에서는 사용금지("csubj"를 제외시켜야함)
is_subject_deps = [ "nsubj", "nsubjpass", "csubj"]

all_nouchunk_types = {
    "subclause_noun", "to.R_noun", "R.ing_ger_noun"
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
        t["role3"] = None


        # role 추론
        role1 = guess_role(t, tokens)
        if role1:
            t["role1"] = role1  # combine에서 쓰일 수 있음
            t["role2"] = role1  
            t["role3"] = role1  


    # 'name'과 같은 동사가 있는 SVOC구조에서 목적보어를 잘못 태깅하는 것 보정 함수
    result = tokens  # 기존 tokens을 수정하며 계속 사용
    result = assign_noun_complement_for_SVOC_noun_only(result)

    # ✅ 보어 기반 object 복구 자동 적용
    result = repair_object_from_complement(result)        


######################################## 신경을 써야할 특별예외처리 부분 ###################################

    ## 특별예외 : 계층발생 ccomp의 자식이 to부정사이고, to부정사의 주체인 앞단어를 nsubj로 태깅하는데,
    #            nsubj가 덩어리요소 시작단어가 되버리는 경우 nsubj(you)를 object로 입력,
    #            to를 noun object complement로 입력 (예문 : I want you to succeed.)

    for t in tokens:
        if t.get("dep") == "ccomp":
            #이경우 ccomp의 자식은 to앞단어(nsubj), to(TO) 모두 ccomp를 head로 본다.
            children = [child for child in tokens if child.get("head_idx") == t["idx"]]
            nsubj_child = next((child for child in children if child.get("dep") == "nsubj"), None)
            to_child = next((child for child in children if child.get("tag") == "TO"), None)

            if nsubj_child and to_child:
                print(f"[DEBUG] ccomp '{t['text']}' has nsubj '{nsubj_child['text']}' + to '{to_child['text']}'")
                nsubj_child["role1"] = "object"
                to_child["role1"] = "noun object complement"
    # assign_level_trigger_ranges에서는 you와 to의 레벨값을 보정함.

#######################################################################################################

    return result



# role 추론 함수
def guess_role(t, all_tokens=None):  # all_tokens 추가 필요
    dep = t.get("dep")
    pos = t["pos"]
    head_idx = t.get("head_idx")

    # ✅ Subject
    if dep in {"nsubj", "nsubjpass"}:
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

            # 1️⃣ 먼저: nsubj 먼저 찾기
            found_object = False
            for t in parsed:
                if t.get("dep") == "nsubj" and t.get("idx") in complement_children:
                    t["role1"] = "object"
                    found_object = True
                    break  # ✅ 단 1회만 보정

            # 2️⃣ 그 다음: compound 찾기 (nsubj 없을 경우만)
            if not found_object:
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
                    compound_candidates.sort(key=lambda x: x["idx"])
                    compound_candidates[0]["role1"] = "object"

    return parsed



# combine 추론 함수
def guess_combine(token, all_tokens):
    token_role1 = token.get("role1")
    token_idx = token.get("idx")
    combine = []

    token_current_level = token.get("level")
    token_head_idx = token.get("head_idx")

    # ✅ Verb → object / complement (SVO, SVC)
    if token_role1 == "verb":
        for t in all_tokens:
            if t.get("idx", -1) <= token_idx: # 이전 토큰이면 continue(다음 토큰부터 찾음)
                continue
            t_head_idx = t.get("head_idx")
            t_head_token = next((x for x in all_tokens if x.get("idx") == t_head_idx), None)
            t_head2_idx = t_head_token.get("head_idx") if t_head_token else None
            t_level = t.get("level")

            if (
                t_head_idx == token_idx or t_head2_idx == token_idx
                and t["idx"] > token_idx  # 🔧 오른쪽 방향 연결만 허용
                and int(t_level) == token_current_level
            ):
                r = t.get("role1")
                if r in [
                    "object",
                    "indirect object",
                    "direct object",
                    "noun subject complement",
                    "adjective subject complement",
#                    "noun object complement",
#                    "adjective object complement"  # 🔧 보어도 연결되게!
                ]:
                    combine.append({"text": t["text"], "role1": r, "idx": t["idx"]})

                    # ✅ 보완: indirect object가 자식 갖고 있으면 그 중 direct object도 연결
                    # ♥♥♥ 이 보완함수를 적용해야 하는 문장을 못찾겠음..
                    if r == "indirect object":
                        children = [c for c in all_tokens if c.get("head_idx") == t["idx"]]
                        for c in children:
                            if (
                                c.get("role1") in ["direct object", "object"]
                                and c["idx"] > t["idx"]  # 🔧 핵심 추가
                            ):
                                combine.append({"text": c["text"], "role1": c["role1"], "idx": c["idx"]})

                    break

    # ✅ Indirect object / object → direct object (SVOO 구조)
    if token_role1 in ("indirect object", "object"):

        for t in all_tokens:
            if t.get("idx", -1) <= token_idx:
                continue
            t_head_idx = t.get("head_idx")
            t_head_token = next((x for x in all_tokens if x.get("idx") == t_head_idx), None)
            t_head2_idx = t_head_token.get("head_idx") if t_head_token else None
            t_level = t.get("level")

            if (
                t.get("role1") == "direct object" and
                t.get("idx") > token_idx and
                int(t_level) == token_current_level
            ):

                if t_head_idx == token_head_idx or t_head2_idx == token_head_idx:
                    combine.append({
                        "text": t["text"], "role1": "direct object", "idx": t["idx"]
                    })

    # ✅ Object → object complement (SVOC 구조)
    if token_role1 == "object":
        for t in all_tokens:
            t_role1 = t.get("role1") or ""
            t_level = t.get("level")
            if t_role1 in ("noun object complement", "adjective object complement"):
                t_head = t.get("head_idx")
                token_head = token.get("head_idx")
                if (
                    token_idx in [c.get("idx") for c in all_tokens if c.get("head_idx") == t.get("idx")]
                    or (t_head is not None and t_head == token_head)
                ):
                    combine.append({"text": t["text"], "role1": t["role1"], "idx": t["idx"]})
                    continue

                # 🔹 추가 연결 조건: 보어가 object보다 뒤에 있고, head는 동일한 동사
                if (
                    t["idx"] > token_idx and
                    t.get("dep") in {"advcl", "oprd", "xcomp", "ccomp"} and
                    t.get("pos") == "ADJ" and
                    t.get("head_idx") == token.get("head_idx") and
                    int(t_level) == token_current_level
                ):
                    combine.append({"text": t["text"], "role1": t["role1"], "idx": t["idx"]})

    # ✅ Preposition → prepositional object
    if token_role1 == "preposition":
        for t in all_tokens:
            t_level = t.get("level")
            if (
                t.get("role1") == "prepositional object" and t.get("head_idx") == token_idx
                and int(t_level) == token_current_level
            ):
                print(f"[DEBUG] prepositional object t.level={t.get('level')}, token.level={token_current_level}")
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
            t_head_idx = t.get("head_idx")
            t_head_token = next((tok for tok in all_tokens if tok["idx"] == t_head_idx), None)

            # ✅ head_token 다음에서 "to" 찾기
            to_token = next(
                (
                    tok for tok in all_tokens
                    if tok["text"].lower() == "to" and tok["idx"] > t_head_token["idx"]
                ),
                None
            )

            # ✅ 이 token이 그 "to"일 경우만 연결
            if to_token and to_token["idx"] == token_idx:
                combine.append({"text": t["text"], "role1": t["role1"], "idx": t["idx"]})

    # ✅ combine 있을 경우만 반환
    return combine if combine else None


def assign_level_trigger_ranges(parsed):
    """
    종속절을 담당하는 dep (relcl, acl, advcl, ccomp, xcomp)에 따라
    해당 절 범위에 level 값을 부여한다.
    
    - relcl, advcl, ccomp, xcomp: 해당 토큰 + children → 범위 계산
    - acl: 해당 토큰부터 children 포함하여 범위 계산 (자기자신이 연결어)
    
    그리고 마지막에 level=None인 토큰들에 대해 level=0을 부여한다.
    """

    clause_units = []  # 절 정보 리스트
    current_level = 1  # 시작은 1부터 (0은 최상위 절용)
    reset_after_root = False  # ✅ ROOT 이후 레벨 초기화 플래그
    prev_clause_indices = set()  # 이전 절 인덱스 저장용

    for token in parsed:
        dep = token.get("dep")

        if dep == "root":
            reset_after_root = True
            continue  # ROOT 자체는 level 트리거 아님

        if reset_after_root:
            current_level = 1
            reset_after_root = False

        if dep not in level_trigger_deps:
            continue
        
        if not is_valid_clause_trigger(token):
            continue

        

        all_clause_indices = []  # 절 단위 인덱스 리스트들을 모아둠
        token_idx = token["idx"]
        clause_tokens = [token]  # 시작은 자기 자신 포함

        # ✅ children도 절 범위에 포함
        children = [t for t in parsed if t["head_idx"] == token_idx]
        clause_tokens.extend(children)

        clause_tokens = [token] + children
        clause_indices = sorted([t["idx"] for t in clause_tokens])
        
        all_clause_indices.append(clause_indices)
    
        print(f"[DEBUG] {all_clause_indices}")

        #is_nested = any(
        #    prev["indices"]
        #    and prev["indices"][0] < clause_indices[0] and prev["indices"][-1] > clause_indices[-1]
        #    for prev in clause_units[:-1]  # 자기 자신 제외
        #)

        for prev_unit in clause_units:
            prev_indices = prev_unit["indices"]
            overlap = set(clause_indices) & set(prev_indices)
            if overlap:
                # 우선순위 판단: 누가 먼저 시작했는지
                if clause_indices[0] < prev_indices[0]:
                    # 현재 clause가 먼저니까, 현재 clause에서 중복 제거
                    clause_tokens = [t for t in clause_tokens if t["idx"] not in overlap]
                    clause_indices = sorted([t["idx"] for t in clause_tokens])
                else:
                    # 이전 clause에서 중복 제거
                    prev_unit["tokens"] = [t for t in prev_unit["tokens"] if t["idx"] not in overlap]
                    prev_unit["indices"] = sorted([t["idx"] for t in prev_unit["tokens"]])

        clause_indices = sorted([t["idx"] for t in clause_tokens])

        # ✅ 절 범위 시작 ~ 끝 계산
        if not clause_tokens:
            continue  # 혹시 다 지워졌으면 skip
        start_idx = min(t["idx"] for t in clause_tokens)
        end_idx = max(t["idx"] for t in clause_tokens)

        clause_units.append({
            "indices": clause_indices,
            "tokens": clause_tokens,
            "connector": token,
        })

        print(f"[DEBUG] {all_clause_indices}")
        print(f"[DEBUG 시작 끝] {start_idx} {end_idx}")

        clause_indices = sorted([t["idx"] for t in clause_tokens])
        clause_indices_set = set(clause_indices)

        # ✅ level 부여
        for t in parsed:
            if start_idx <= t["idx"] <= end_idx:
                if (
                    t.get("level") is None
                    #or t["idx"] in prev_clause_indices  # 이전 절과 겹치는 경우만 덮어쓰기 허용
                    #or not is_nested
                ):
                    t["level"] = current_level
        
        prev_clause_indices = clause_indices_set

######################################## 신경을 써야할 특별예외처리 부분 ###################################

    ## 특별예외 : 계층발생 ccomp의 자식이 to부정사이고, to부정사의 주체인 앞단어를 nsubj로 태깅하는데,
    #            nsubj가 덩어리요소 시작단어가 되버리는 경우 그 뒤 to를 시작단어(.5)로 수정하고
    #            you는(.5)를 없앰 (예문 : I want you to succeed.)

        # ✅ 단어덩어리 맨 앞 토큰 찾기
        sorted_clause = sorted(clause_tokens, key=lambda x: x["idx"])
        first_token = sorted_clause[0]

        # 🔥 단어덩어리 맨 앞 단어가 nsubj인지 체크
        if first_token.get("dep") == "nsubj":
            to_token = next((child for child in children if child.get("tag") == "TO"), None)
            if to_token:
                to_head_idx = to_token.get("head_idx")
                to_head_token = next((t for t in parsed if t["idx"] == to_head_idx), None)

                if to_head_token and to_head_token.get('dep') == "ccomp":
                    # 🎯 핵심: TO가 연결된 ccomp 절이면 레벨 설정
                    to_token["level"] = current_level - 0.5
                    first_token["level"] = current_level - 1
                else:
                    # 🎯 TO 없거나 조건 불충족 시, nsubj만 .5 레벨
                    first_token["level"] = current_level - 0.5
            else:
                first_token["level"] = current_level - 0.5

            current_level += 1
            continue
    # rule_base_parse() 함수에서는 you와 to의 role1을 입력함.

#######################################################################################################

        # ✅ 연결어에는 .5 추가
        if dep == "acl":
            token["level"] = current_level - 0.5  # 연결어는 바로 이전 절에서 이어짐
        else:
            # 연결어 후보: 절 범위 앞 단어 중 연결사 역할
            connector = min(clause_tokens, key=lambda x: x["idx"])
            connector["level"] = current_level - 0.5

        current_level += 1

    for i in range(len(clause_units) - 1):
        unit1 = clause_units[i]
        unit2 = clause_units[i + 1]

        first = unit1["indices"]
        second = unit2["indices"]

        if not first or not second:
            continue

        if second[0] < first[0] and second[-1] > first[-1]:
            # 안은 절 +1
            for t in unit1["tokens"]:
                if t.get("level") is not None:
                    t["level"] += 1
                    print(f"[DEBUG 디버그11111] {current_level}")
            # 안긴 절 -1 (겹치는 것 빼고)
            for t in unit2["tokens"]:
                if t["idx"] not in first and t.get("level") is not None:
                    t["level"] -= 1
                    print(f"[DEBUG 디버그22222] {current_level}")


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
    예문) She is certain that he will arrive on time.
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

    return parsed


def get_subclause_verbals_type(token, all_tokens):

    # 1️⃣ 종속절 (Subordinate Clause)
    if (
        token.get("pos") == "VERB" and
        token.get("dep") in {"ccomp", "xcomp", "advcl", "csubj"}
    ):
        children = [t for t in all_tokens if t.get("head_idx") == token["idx"]]
        has_sconj_marker = any(
            c.get("dep") in {"mark", "advmod"} and c.get("pos") == "SCONJ" and
            c.get("text", "").lower() in {
                "that", "if", "whether", "because", "although", "since",
                "when", "unless", "though", "as"
            }
            for c in children
        )
        if has_sconj_marker:
            return "subordinate_clause"

    # 2️⃣ to 부정사
    if (
        token.get("pos") == "PART" and token.get("tag") == "TO" and token.get("dep") == "aux"
        and token.get("lemma", "").lower() == "to"
    ):
        to_idx = token["idx"]
        verb_after = next(
            (t for t in all_tokens if t["idx"] > to_idx and t.get("pos") == "VERB" and t.get("tag") == "VB"),
            None
        )
        if verb_after:
            return "to_infinitive"

    # 3️⃣ bare infinitive (TO 없이 동사 원형)
    if (
        token.get("pos") == "VERB" and
        token.get("tag") == "VB" and
        not any(
            t.get("idx") == token["idx"] - 1 and
            t.get("text", "").lower() == "to" and
            t.get("tag") == "TO"
            for t in all_tokens
        )
    ):
        return "bare_infinitive"
    
    # 4️⃣ 동명사
    if (
        token.get("morph", {}).get("VerbForm") == "Ger" or
        (token.get("tag") == "VBG" and token.get("text", "").lower().endswith("ing"))
        # token.get("dep") in {"nsubj", "dobj", "obj", "pobj", "attr"}
    ):
        return "gerund"  # 동명사

    # 5️⃣ 현재분사
    if token.get("tag") == "VBG":
        verb_form = token.get("morph", {}).get("VerbForm")
        if verb_form != "Ger":
            if token.get("pos") == "VERB" and token.get("dep") in {
                "amod", "acl", "advcl", "xcomp", "ccomp", "conj"
            }:
                return "present_participle"

    # 6️⃣ 과거분사
    if token.get("tag") == "VBN" and token.get("pos") == "VERB":
        verb_form = token.get("morph", {}).get("VerbForm")
        if not verb_form or verb_form == "Part":
            return "past_participle"

    # 7️⃣ reduced clause (분사구문)
    if (
        token.get("pos") == "VERB" and
        token.get("dep") in {"advcl", "amod"} and
        token.get("tag") in {"VBG", "VBN"}
    ):
        return "reduced_clause"

    return None


def get_chunks_partofspeech(token, all_tokens):

    form_type = get_subclause_verbals_type(token, all_tokens)
    dep = token.get("dep")

    if form_type == "subordinate_clause" and dep in {"nsubj", "csubj", "obj", "dobj"}:
        children = [t for t in all_tokens if t.get("head_idx") == token["idx"]]
        has_noun_sconj = any(
            c.get("pos") == "SCONJ" and
            c.get("dep") == "mark" and
            c.get("tag") == "IN" and
            c.get("text", "").lower() in {"that", "if", "whether"}  # 명사절 전용 SCONJ
            for c in children
        )
        if has_noun_sconj:
            return "subclause_noun"
        
    if form_type == "subordinate_clause":
        children = [t for t in all_tokens if t.get("head_idx") == token["idx"]]
        has_adv_sconj = any(
            c.get("pos") == "SCONJ" and
            c.get("dep") in {"mark", "advmod"} and
            c.get("tag") in {"IN", "WRB"} and
            c.get("text", "").lower() in {
                "because", "since", "although", "when", "while", "if", "unless", "as", "though"
            }
            for c in children
        )
        if has_adv_sconj:
            return "subclause_adverb"


    if form_type == "to_infinitive":
        token["role2"] = "to infinitive"

        head_idx = token.get("head_idx")
        head_token = next((t for t in all_tokens if t["idx"] == head_idx), None)
        head_dep = head_token.get("dep") if head_token else None

        if head_dep in {"csubj"}:
            return "to.R_noun"
        elif head_dep in {"xcomp", "ccomp"}:
            return "to.R_noun.adj_dontcare"
        elif head_dep in {"relcl"}:
            return "to.R_adjective"
        elif head_dep in {"advcl"}:
            return "to.R_adverb"


    if form_type == "gerund":
        token["role2"] = "gerund"
        if token.get("dep") in {"nsubj", "csubj", "obj", "dobj", "pobj", "attr"}:
            return "R.ing_ger_noun"

    return None  # 해당사항 없으면 None


def assign_chunk_roles_and_drawsymbols(parsed):

    all_subject_complements = {
        "noun subject_complement", "adjective subject_complement"
    }

    line_length = memory["sentence_length"]
    symbols_by_level = memory["symbols_by_level"]

    # 계층시작요소(level x.5단어)가 아니면 루프 빠져 나감
    for token in parsed:
        level = token.get("level")
        if not (isinstance(level, float) and level % 1 == 0.5):
            continue

        #계층시작요소의 헤드 값이 없으면 루프 빠져 나감
        head_idx = token.get("head_idx")
        head_token = next((t for t in parsed if t["idx"] == head_idx), None)
        if not head_token:
            continue

        chunks_pos = get_chunks_partofspeech(token, parsed)

        token_dep = token.get("dep")
        head_dep = head_token.get("dep")

        # 부사덩어리 먼저 확정 : 덩어리요소 첫단어의 head의 dep가 advcl이고,
        # chunks_pos == "subclause_adverb"이면 role3에 'chunk_adverb_modifier'값 입력

        if (chunks_pos == "subclause_adverb" and token_dep == "advcl" or head_dep == "advcl"):
            token["role3"] = "chunk_adverb_modifier"
            continue  # ✅ 부사절이면 명사절 분기로 건너뜀

        chunks_partofspeech = get_chunks_partofspeech(token, parsed)

        # 주어 명사덩어리 그다음 확정 : 덩어리요소 첫단어의 head의 dep가 csubj, nsubj, nsubjpass이고,
        # is_nounchunk_trigger() 함수에 걸리면 role3에 'chunk_subject'값 입력
        if (chunks_partofspeech and (token_dep in is_subject_deps) or (head_dep in is_subject_deps)):
            print(f"[DEBUG-chunks_partofspeech 02 in assign_chunk_roles_and_drawsymbols] {chunks_partofspeech}")
            token["role3"] = "chunk_subject"
            continue

        # 명사덩어리 판단 : '계층시작요소' 또는 '계층시작요소의 헤드'의 dep가(ccomp, xcomp) 이고, 
        # 계층시작요소가 is_nounchunk_trigger에 걸리면,
        print(f"[DEBUG-chunks_partofspeech 01 in assign_chunk_roles_and_drawsymbols {chunks_partofspeech}")

        if (
            (token_dep in {"ccomp", "xcomp"} or head_dep in {"ccomp", "xcomp"})
            and chunks_partofspeech
        ):
            if chunks_partofspeech == "to.R_noun":  
                token["role2"] = "to infinitive"

            if (
                (token_dep in {"xcomp"} or head_dep in {"xcomp"})
                and chunks_partofspeech == "R.ing_ger_noun"
            ):
                token["role2"] = "gerund"   # ☜ 확인필요 아래 build를 gerund로 저장함
                                            # To be honest helps build trust.

            # 계층시작요소의 유효한 head 찾아서 head값이 없으면 루프 빠져나감
            # to부정사(to infinitive)인 경우만 head의 head로 타고 올라가기
            head2_token = (
                next((t for t in parsed if t["idx"] == head_token.get("head_idx")), None)
                if chunks_partofspeech in {"to.R_noun", "subclause_noun"}
                else head_token
            )
            if not head2_token:
                continue

            # 계층시작요소 head의 head인 상위동사(head2) lemma값 저장
            head2_lemma = head2_token.get("lemma", "")

            # 1) 명사덩어리 확정후 상위동사가 be동사와 LinkingVerbs인데,
            # 상위동사(현토큰 헤드의 헤드)가 이미 명사보어, 형용사보어를 가지고 있지 않을경우 보어 확정
            # 명사덩어리 첫단어의 role1에 'noun subject complement'(명사주어보어)값 저장
            if (
                head2_lemma in beVerbs or head2_lemma in notbeLinkingVerbs_onlySVC
            ) and not any(
                c.get("role1") in {"all_subject_complements"}
                for c in head2_token.get("combine", [])
            ):
                token["role1"] = "noun subject complement"

            # 2) 상위동사가 dativeVerbs일때 상위동사level 단어들의 role1에 objedct, indirect object가 있으면
            # 명사덩어리 첫단어의 role1에 'direct object'(직접목적어)값 저장
            # 아니면 role1에 'object'(목적어)값 저장
            elif head2_lemma in dativeVerbs:
                current_level = int(level)  # x.5 -> x
                # 현재 레벨의 토큰들
                level_tokens = [t for t in parsed if int(t.get("level", -1)) == current_level]
                has_obj_or_iobj = any(
                    t.get("role1") in {"object", "indirect object"} for t in level_tokens
                )
                if has_obj_or_iobj:
                    token["role1"] = "direct object"
                else:
                    token["role1"] = "object"

            # 앞 모든 조건에 안걸리면 명사덩어리 첫단어의 rele1에 'object'(목적어) 값 저장
            else:
                token["role3"] = "chunk_not_decide"
                token["role1"] = "object"


        # ✅ # 현토큰의 head의 children들 모음 (끝 토큰 찾기 + 시작 토큰 info)
        children_tokens = [child for child in parsed if child.get("head_idx") == head_idx]
        children_tokens.append(head_token)              # 현토큰의 head token까지 병합
        children_tokens.sort(key=lambda x: x["idx"])    # 단어들의 순서를 왼쪽부터 정렬함
        end_token = children_tokens[-1]

        # 끝 토큰이 구두점(. ! ?)이면 그 앞 토큰 사용
        if (
            end_token.get("pos") == "PUNCT" and
            end_token.get("text") in {".", "!", "?"} and
            len(children_tokens) >= 2
        ):
            end_token = children_tokens[-2]

        start_idx = token["idx"]
        end_idx = end_token["idx"]
        end_idx_adjusted = end_idx + len(end_token.get("text", "")) - 1 # 끝단어 끝글자 인덱스 계산
        int_level = int(level) # .5요소이지만 덩어리 끝표시를 상위계층에 맞추어 그려야 하므로 소수점(.5)버림

        role1 = token.get("role1")
        line = symbols_by_level.setdefault(int_level, [" " for _ in range(line_length)])

        chunk_end_mark = None

        # ✅ 기본 심볼
        if role1 in {"noun subject complement", "object", "indirect object", "direct object",
                     "noun object complement"}:
            chunk_end_mark = "]"

#        if 0 <= start_idx < line_length:
#            line[start_idx] = left
        if chunk_end_mark and (0 <= end_idx_adjusted < line_length) and token.get("pos") != "VERB":
            line[end_idx_adjusted] = chunk_end_mark

        # ✅ to infinitive → to.o...R
        if token.get("role2") == "to infinitive":
            verb_token = next(
                (t for t in parsed
                 if t["idx"] > start_idx and
                    int(t.get("level", 0)) == int_level + 1 and
                    t.get("pos") == "VERB"),
                None
            )
            if verb_token:
                verb_idx = verb_token["idx"]
                verb_end = verb_idx + len(verb_token["text"]) - 1
                line2 = symbols_by_level.setdefault(int_level + 1, [" " for _ in range(line_length)])
                if 0 <= start_idx < line_length: line2[start_idx] = "t"
                if 0 <= start_idx + 1 < line_length: line2[start_idx + 1] = "o"
                for i in range(start_idx + 2, verb_end):
                    if 0 <= i < line_length and line2[i] == " ":
                        line2[i] = "."
                if 0 <= verb_end < line_length: line2[verb_end] = "R"

        # ✅ gerund → R...ing
        if token.get("role2") == "gerund":
            verb_token = next(
                (t for t in parsed
                 if t["idx"] >= start_idx
                    and (t.get("level") == level or int(t.get("level", 0)) == int_level + 1)
                    and t.get("pos") == "VERB"),
                None
            )
            if verb_token:
                verb_idx = verb_token["idx"]
                verb_end = verb_idx + len(verb_token["text"]) - 1
                line2 = symbols_by_level.setdefault(int_level + 1, [" " for _ in range(line_length)])
                if 0 <= start_idx < line_length: line2[start_idx] = "R"
                for i in range(start_idx + 1, verb_end-2):
                    if 0 <= i < line_length and line2[i] == " ":
                        line2[i] = "."
                if 0 <= verb_end < line_length: line2[verb_end-2] = "i"
                if 0 <= verb_end < line_length: line2[verb_end-1] = "n"
                if 0 <= verb_end < line_length: line2[verb_end] = "g"

    return parsed
    

def apply_subject_adverb_chunk_range_symbol(parsed):
    """
    role3=chunk_subject인 토큰을 기준으로
    해당 절(start_idx ~ end_idx) 범위에 [ ] 심볼 부여
    """
    line_length = memory["sentence_length"]
    symbols_by_level = memory["symbols_by_level"]

    for token in parsed:
        role3 = token.get("role3")
        if not role3:
            continue

        level = token.get("level")
        if level is None:
            continue

        line = symbols_by_level.setdefault(int(level), [" " for _ in range(line_length)])
        chunks_partofspeech = get_chunks_partofspeech(token, parsed)

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


        # 동명사의 경우 gerund의 헤드의 자식을 이용해야하는데 gerund 범위 밖의 단어까지 포함되버림
        # 이 부분은 그 경우의 보정임. 예문) Watching movies affects my sleep.
        if role3 == "chunk_subject" and chunks_partofspeech == "R.ing_ger_noun":
            for i, child in enumerate(children_tokens):
                if child.get("pos") == "VERB" and child["idx"] > token["idx"]:
                    if i > 0:
                        end_token = children_tokens[i - 1]
                    break

        # 동명사덩어리가 주어인 경우 덩어리 끝이 마침표 나올일 없음(이 소스는 필요없어 보임)
        # if end_token.get("pos") == "PUNCT" and len(children_tokens) >= 2:
        #    end_token = children_tokens[children_tokens.index(end_token) - 1]

        end_idx = end_token["idx"]
        end_idx_adjusted = end_idx + len(end_token["text"]) - 1

        # ✅ role3에 따라 심볼 다르게
        if role3 == "chunk_subject":
            left, right = "[", "]"
        elif role3 == "chunk_adverb_modifier":
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
        if role2 not in {
            "object", "direct object",
             "noun subject complement", "noun object complement"
        }:
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

    verb_attr = memory["symbols_all"]["verb_attr"]

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

    print(chains)

# ◎ GPT 프롬프트 처리 함수
def spacy_parsing_backgpt(sentence: str, force_gpt: bool = False):

#    memory["used_gpt"] = False  # ✅ 기본값: GPT 미사용
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


    # level 분기 전파
    parsed = assign_level_trigger_ranges(parsed)

    # ✅ 요기! 모든 보정 끝난 후에 combine 추론
    for t in parsed:
        combine = guess_combine(t, parsed)
        if combine:
            t["combine"] = combine

    # 조건: 규칙 기반 실패하거나, 강제로 GPT 사용 요청
    if not parsed or force_gpt:
        memory["used_gpt"] = True  # ✅ GPT fallback 사용된 경우
        # GPT 파싱 호출
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

    assign_chunk_roles_and_drawsymbols(parsed)  # ★★★★ 위의 assign_level_trigger_ranges() 함수 위로 갈 수 없다.
                                                # 그래서 guess_combine_second()를 한번 더 호출한다.

    # ✅ 📍 level 보정: prep-pobj 레벨 통일
    parsed = repair_level_within_prepositional_phrases(parsed)

    parsed = guess_combine_second(parsed)

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
Each item must have: idx, text, role1, role2, role3, and optionally combine/level.

If unsure, return best-guess. Do not return explanations, just the JSON.
"""
    return prompt.strip()


# ◎ 저장공간 초기화
def init_memorys (sentence: str):
#    memory["characters"] = list(sentence)        # characters에 sentence의 글자 한글자씩 채우기
    memory["symbols_by_level"] = {}  # 문장마다 새로 초기화
    memory["sentence_length"] = len(sentence)  # 도식 길이 추적용 (줄 길이 통일)


def lookup_symbol(name):
    name = name.lower()
    for symbol_category in symbols_all.values():
        for key, value in symbol_category.items():
            if key.lower() == name:
                return value
    return None

# ◎ symbols 메모리에 심볼들 저장하기
def apply_symbols(parsed):
    symbols_by_level = memory["symbols_by_level"]
    line_length = memory["sentence_length"]

    for item in parsed:
        idx = item.get("idx", -1)
        role1 = item.get("role1", "") or ""
        role2 = item.get("role2", "") or ""
        level = item.get("level")

        if idx < 0 or level is None:
            continue

        symbol1 = lookup_symbol(role1)
        symbol2 = lookup_symbol(role2)

        # ✅ 1. role1: 정수 레벨에만 찍기
        levels_role1 = [int(level)]  # <--- 여기 수정
        for lvl in levels_role1:
            line = symbols_by_level.setdefault(lvl, [" " for _ in range(line_length)])
            if 0 <= idx < len(line) and line[idx] == " " and symbol1:
                line[idx] = symbol1

        # ✅ 2. role2: (0.5 레벨 단어에만)
        if isinstance(level, float) and (level % 1 == 0.5):
            lvl_role2 = int(level) + 1
            line2 = symbols_by_level.setdefault(lvl_role2, [" " for _ in range(line_length)])
            if 0 <= idx < len(line2) and line2[idx] == " " and symbol2:
                line2[idx] = symbol2

    # ⬇️ combine 연결선을 _ 로 그려줌!
    for item in parsed:
        combine = item.get("combine")
        level = item.get("level")
        idx1 = item.get("idx")

        if not combine or level is None:
            continue

        for comb in combine:
            idx2 = comb.get("idx")
            if idx2 is None:
                continue

            lvl = int(level + 0.5)

            line = symbols_by_level.setdefault(lvl, [" " for _ in range(line_length)])
            start = min(idx1, idx2)
            end = max(idx1, idx2)

            for i in range(start + 1, end):
                if line[i] == " ":
                    line[i] = "_"

    return parsed


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


def guess_combine_second(parsed):
    for token in parsed:
        combine = guess_combine(token, parsed)
        if combine:
            token["combine"] = combine
    return parsed


def t(sentence: str):
    print(f"\n📘 Sentence: {sentence}")

    # ✅ 메모리 먼저 초기화 (문장 길이 기반 설정 포함)
    init_memorys(sentence)

    # ✅ spaCy 파싱 + 역할 분석
    parsed = spacy_parsing_backgpt(sentence)
    memory["parsed"] = parsed

#    if memory.get("used_gpt"):
#        print("⚠️ GPT가 파싱에 개입했음 (속도 느릴 수 있음)")
#    else:
#        print("✅ spaCy 규칙 기반으로 파싱 완료")

   # NounChunk_combine_apply_to_upverb(parsed)
    apply_symbols(parsed)
    apply_subject_adverb_chunk_range_symbol(parsed)
    draw_dot_bridge_across_verb_group(parsed)

    # ✅ morph 상세 출력
    print("\n📊 Full Token Info with Annotations:")
    print(nlp.path)
    doc = nlp(sentence)
    for token in doc:
        morph = token.morph.to_dict()
        idx = token.idx
        text = token.text
        role1 = next((t.get("role1") for t in parsed if t["idx"] == idx), None)
        role2 = next((t.get("role2") for t in parsed if t["idx"] == idx), None)
        role3 = next((t.get("role3") for t in parsed if t["idx"] == idx), None)

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

        print(f"● idx({idx}), text({text}), role1({role1}), role2({role2}), role3({role3}), combine({combine_str})")
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
    chunk_info_list = assign_chunk_role(parsed)
    NounChunk_combine_apply_to_upverb(parsed)
    apply_subject_adverb_chunk_range_symbol(parsed)
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
    "assign_level_trigger_ranges",
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
    apply_symbols(parsed)
    apply_subject_adverb_chunk_range_symbol(parsed)
    draw_dot_bridge_across_verb_group(parsed)
    return {"sentence": request.sentence,
            "diagramming": symbols_to_diagram(request.sentence),
            "verb_attribute": memory.get("verb_attribute", {}),
            "used_gpt": memory.get("used_gpt", False)  # ✅ 결과 포함
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
