# subcode.py

from analyzer_utils import memory, role_to_symbol

def apply_symbols(parsed_result):
    text = ''.join(memory["characters"]).lower()

    for item in parsed_result:
        word = item["word"].lower()
        role = item["role"].lower()
        symbol = role_to_symbol.get(role)

        if symbol:
            index = text.find(word)
            if index != -1:
                for i in range(len(word)):
                    memory["symbols"][index + i] = symbol

def generate_diagram():
    output = ""
    for char, symbol in zip(memory["characters"], memory["symbols"]):
        if symbol != " ":
            output += f"{symbol}{char}"
        else:
            output += f" {char}"
    return output
