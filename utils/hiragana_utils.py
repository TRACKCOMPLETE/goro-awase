import jaconv
import re
import random
import MeCab
from itertools import permutations

tagger = MeCab.Tagger()

def get_first_hiragana(word: str) -> str:
    if len(word) == 0:
        return "N"

    first_char = word[0]
    if re.match(r'[\u3040-\u309F]', first_char):  # ひらがな
        return first_char
    elif re.match(r'[\u30A0-\u30FF]', first_char):  # カタカナ
        return jaconv.kata2hira(first_char)
    elif re.match(r'[\u4E00-\u9FFF]', first_char):  # 漢字
        parsed = tagger.parse(word).split('\t')
        return jaconv.kata2hira(parsed[1][0])
    else:
        return "?"
    
def get_first_hiragana_of_words(words: list[str]) -> list[str]:
    results = []
    for word in words:
        hira = get_first_hiragana(word)
        if re.match(r'[\u3040-\u309F]', hira):
            results.append(hira)
    return results

def random_hira(n: int) -> list[str]:
    gojuon = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわ"
    return [random.choice(gojuon) for _ in range(n)]

def generate_permutations(hira_list: list[str]) -> list[str]:
    perms = set(''.join(p) for p in permutations(hira_list))
    return sorted(perms)