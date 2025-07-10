# text_filters.py

import MeCab

tagger = MeCab.Tagger("-Ochasen")

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

def is_all_nouns(parsed: list[str]) -> bool:
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

def should_exclude(words: list[str]) -> list[bool]:
    results = []
    for word in words:
        exclude = contains_symbol_by_pos(word) or contains_no_noun(word) or is_all_nouns(word)
        results.append(exclude)
    return results
