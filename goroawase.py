# -*- coding: utf-8 -*-
# pip install jaconv mecab-python3 unidic-lite
# pip install transformers fugashi ipadic torch

import jaconv
import MeCab
import unidic_lite
import re
import torch
import random
from itertools import permutations
from transformers import BertJapaneseTokenizer, BertForMaskedLM

tagger = MeCab.Tagger()
tagger_wakati = MeCab.Tagger("-Owakati")


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

def generate_permutations(hira_list: list[str]) -> list[str]:
    perms = set(''.join(p) for p in permutations(hira_list))
    return sorted(perms)

def contains_interjection_or_adverb(parsed: list[str]) -> bool:
    """
    感動詞（間投詞）または副詞が含まれているかチェック。
    含まれていたらTrue（アウト）
    """
    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        pos = cols[4].split("-")[0]
        if pos in {"感動詞", "副詞"}:
            return True
    return False


def check_positions(parsed: list[str]) -> bool:
    """
    助詞・接続詞の位置、孤立感動詞、文末副詞などを一括チェック
    """
    pos_list = []
    tokens = []

    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        surface = cols[0]
        pos = cols[4].split("-")[0]
        pos_list.append(pos)
        tokens.append((surface, pos))

    # 助詞の位置チェック（文頭・連続禁止）
    prev_pos = None
    for pos in pos_list:
        if pos == "助詞":
            if prev_pos is None or prev_pos == "助詞":
                return False
        prev_pos = pos

    # 接続詞の位置チェック（文頭・文末・連続禁止）
    for i, pos in enumerate(pos_list):
        if pos == "接続詞":
            if i == 0 or i == len(pos_list) - 1:
                return False
            if pos_list[i - 1] == "接続詞" or pos_list[i + 1] == "接続詞":
                return False

    # 文末が副詞はNG
    if pos_list and pos_list[-1] == "副詞":
        return False

    # 孤立感動詞チェック（感動詞が1語だけで存在する場合NG）
    interjection_count = sum(1 for _, p in tokens if p == "感動詞")
    if interjection_count == 1 and len(tokens) == 1:
        return False

    return True


def check_pos_usage(parsed: list[str]) -> bool:
    """
    動詞活用、助動詞の使い方、助詞の意味的使い方、否定接頭辞などをチェック
    """
    tokens = []
    prev_pos = None
    NEGATIVE_PREFIX_RULES = {
        "不": {"形状詞", "名詞", "形容動詞"},
        "未": {"名詞"},
        "無": {"名詞"},
        "非": {"名詞"},
    }

    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 7:
            continue
        surface = cols[0]
        pos_detail = cols[4]
        pos = pos_detail.split("-")[0]
        conj_form = cols[6]
        tokens.append((surface, pos, pos_detail, conj_form))

    # 動詞の活用形チェック（未然形、仮定形はNG）
    for _, pos, pos_detail, conj_form in tokens:
        if pos == "動詞" and (conj_form.startswith("未然形") or conj_form.startswith("仮定形")):
            return False

    # 助動詞の前が動詞であること
    for i, (_, pos, _, _) in enumerate(tokens):
        if pos == "助動詞":
            if i == 0 or tokens[i - 1][1] != "動詞":
                return False

    # 助詞の意味的使い方チェック
    for i, (surface, pos, _, _) in enumerate(tokens):
        if pos != "助詞":
            continue
        if i == 0 or i == len(tokens) - 1:
            return False  # 文頭文末の助詞はNG
        prev_pos = tokens[i - 1][1]
        next_pos = tokens[i + 1][1]

        if surface == "が":
            if prev_pos != "名詞" or next_pos not in {"動詞", "形容詞", "名詞"}:
                return False
        elif surface == "を":
            if prev_pos != "名詞" or next_pos != "動詞":
                return False
        elif surface == "に":
            if prev_pos != "名詞" or next_pos not in {"動詞", "名詞"}:
                return False
        elif surface == "の":
            if prev_pos != "名詞" or next_pos != "名詞":
                return False
        elif surface == "と":
            if prev_pos != "名詞" or next_pos not in {"名詞", "動詞"}:
                return False
        elif surface == "で":
            if prev_pos != "名詞" or next_pos not in {"動詞", "形容詞"}:
                return False
        else:
            return False  # 未知の助詞はNG

    # 否定接頭辞の使い方チェック
    for i in range(len(tokens) - 1):
        surface1, pos1, pos_detail1, _ = tokens[i]
        surface2, pos2, _, _ = tokens[i + 1]

        if surface1 in NEGATIVE_PREFIX_RULES and pos_detail1.startswith("接頭辞"):
            allowed_pos = NEGATIVE_PREFIX_RULES[surface1]
            if pos2 not in allowed_pos:
                return False
            
    # 接尾辞チェック
    if tokens and tokens[0][2].startswith("接尾辞"):
        return False
    for i in range(len(tokens) - 1):
        _, _, pos_detail, _ = tokens[i]
        _, _, next_pos_detail, _ = tokens[i + 1]
        if pos_detail.startswith("接尾辞") and next_pos_detail.startswith("接尾辞"):
            return False

    # 接頭辞が文末に来るのはNG
    if tokens and tokens[-1][2].startswith("接頭辞"):
        return False

    return True


def validate_parsed_sentence(parsed: list[str]) -> bool:
    if contains_interjection_or_adverb(parsed):
        return False
    if not check_positions(parsed):
        return False
    if not check_pos_usage(parsed):
        return False
    return True

def is_in_counts(word: str, n: int) -> str | None:
    parsed_wakati = tagger_wakati.parse(word).strip().split()
    parsed = tagger.parse(word).strip().split("\n")

    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        pos = cols[4].split("-")[0]
        break
    else:
        return None

    if (
        len(parsed_wakati) <= n
        and validate_parsed_sentence(parsed)
    ):
        return word
    else:
        return None

def check_counts(words: list[str], n: int) -> list[str]:
    return [word for word in words if is_in_counts(word, n)]

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = BertForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese")
model.eval()

def get_bert_score(text: str) -> float:
    tokens = tokenizer.tokenize(text)
    if len(tokens) < 1:
        return -float('inf')

    total_score = 0.0
    with torch.no_grad():
        for i in range(len(tokens)):
            masked_tokens = tokens.copy()
            masked_tokens[i] = "[MASK]"

            input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            input_tensor = torch.tensor([input_ids])

            outputs = model(input_tensor)
            logits = outputs.logits[0]

            mask_index = input_ids.index(tokenizer.mask_token_id)
            target_token_id = tokenizer.convert_tokens_to_ids([tokens[i]])[0]
            log_prob = torch.log_softmax(logits[mask_index], dim=-1)[target_token_id]

            total_score += log_prob.item()
    return total_score

def most_natural_string(candidates):
    valid = [c for c in candidates if len(c) > 0]
    if not valid:
        raise ValueError("候補リストが空")
    return max(valid, key=get_bert_score)

def random_hira(n: int) -> list[str]:
    gojuon = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわ"
    return [random.choice(gojuon) for _ in range(n)]

if __name__ == "__main__":
    # words = ["深層岩", "花崗岩", "閃緑岩", "斑レイ岩", "火山岩", "流紋岩", "安山岩", "玄武岩"]
    # words = ["光", "対立", "こう", "サヤ", "エト", "ルナ", "しらべ", "イリス"]
    # words = ["花海咲季", "月村手毬", "藤田ことね", "有村麻央", "姫崎莉波", "十王星南", "雨夜燕"]

    
    words = random_hira(9)
    print("入力単語(ひらがな):", words)

    hiras = get_first_hiragana_of_words(words)
    print("先頭ひらがな:", hiras)

    pattern = generate_permutations(hiras)
    print("全組み合わせ数:", len(pattern))

    for n in range(2, 5):
        valid = check_counts(pattern, n)
        print(f"単語数 <= {n} の候補数:", len(valid))
        if valid:
            result = most_natural_string(valid)
            print(f"最も自然な文字列: {result}")
            print(tagger.parse(result))
            break

    # print(tagger.parse("たべられます")) # チェック用