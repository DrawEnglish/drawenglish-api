from app.main import gpt_parse

# ✅ 분석 결과를 콘솔에 출력 (디버깅용)
def print_parsed_roles(sentence: str):
    parsed_result = gpt_parse(sentence)

    if not parsed_result:
        print("❌ GPT 응답을 파싱하지 못했습니다.")
        return

    for item in parsed_result:
        word = item.get("word")
        role = item.get("role")
        print(f"{word} - {role}")


if __name__ == "__main__":
    print_parsed_roles("I love you.")
    print_parsed_roles("She gave me a book.")
