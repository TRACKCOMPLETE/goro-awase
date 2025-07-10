# text_filters.py

import MeCab

tagger = MeCab.Tagger()

def contains_symbol_by_pos(text: str) -> bool:
    node = tagger.parseToNode(text)
    while node:
        features = node.feature.split(',')
        if features[0].startswith('記号'):
            return False
        node = node.next
    return True

def contains_no_noun(text: str) -> bool:
    node = tagger.parseToNode(text)
    while node:
        features = node.feature.split(',')
        if features[0] == '名詞':
            return True
        node = node.next
    return False

def is_not_all_nouns(parsed: list[str]) -> bool:
    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        pos = cols[4].split("-")[0]
        if pos != "名詞":
            return True
    return False


def should_pass(words: list[str]) -> list[str]:
    results = []
    for word in words:
        cond1 = contains_symbol_by_pos(word)
        cond2 = contains_no_noun(word)
        parsed = tagger.parse(word).split("\n")
        cond3 = is_not_all_nouns(parsed)

        if cond1 and cond2 and cond3:
            results.append(word)
    return results
