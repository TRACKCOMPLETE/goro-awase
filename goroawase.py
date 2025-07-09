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

def is_valid_inflection(parsed: list[str]) -> bool:
    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 7:
            continue
        pos_detail = cols[4]
        conj_form = cols[6]
        if pos_detail.startswith("動詞") and conj_form.startswith("未然形"):
            return False
    return True

def is_valid_adverb_placement(parsed: list[str]) -> bool:
    morphemes = []
    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        pos = cols[4].split("-")[0]
        morphemes.append(pos)
    if not morphemes:
        return True  # 空なら問題なし
    return morphemes[-1] != "副詞"

def is_no_lone_interjection(parsed: list[str]) -> bool:
    has_interjection = False
    token_count = 0
    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        pos = cols[4].split("-")[0]
        if pos == "感動詞":
            has_interjection = True
        token_count += 1
    return not (has_interjection and token_count == 1)

def is_valid_particle_position(parsed: list[str]) -> bool:
    prev_pos = None
    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        if "\t" not in line:
            continue

        surface, features = line.split("\t", 1)
        pos = features.split(",")[0]

        if pos == "助詞":
            if prev_pos in [None, "助詞"]:  # 文頭 or 助詞連続
                return False
        prev_pos = pos

    return True

def is_valid_particle_semantics(parsed: list[str]) -> bool:
    tokens = []
    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        surface = cols[0]
        pos = cols[4].split("-")[0]
        tokens.append((surface, pos))

    for i, (word, pos) in enumerate(tokens):
        if pos != "助詞":
            continue

        if i == 0 or i == len(tokens) - 1:
            return False  # 助詞が文頭/文末は不自然

        prev_word, prev_pos = tokens[i - 1]
        next_word, next_pos = tokens[i + 1]

        if word == "が":
            if prev_pos != "名詞" or next_pos not in {"動詞", "形容詞", "名詞"}:
                return False
        elif word == "を":
            if prev_pos != "名詞" or next_pos != "動詞":
                return False
        elif word == "に":
            if prev_pos != "名詞" or next_pos not in {"動詞", "名詞"}:
                return False
        elif word == "の":
            if prev_pos != "名詞" or next_pos != "名詞":
                return False
        elif word == "と":
            if prev_pos != "名詞" or next_pos not in {"名詞", "動詞"}:
                return False
        elif word == "で":
            if prev_pos != "名詞" or next_pos not in {"動詞", "形容詞"}:
                return False

    return True

def is_valid_negative_prefix_usage(parsed: list[str]) -> bool:
    NEGATIVE_PREFIX_RULES = {
        "不": {"形状詞", "名詞", "形容動詞"},
        "未": {"名詞"},  # 実際は「未解決」みたいに動詞の名詞化が多い
        "無": {"名詞"},
        "非": {"名詞"},
    }

    for i in range(len(parsed) - 1):
        line1 = parsed[i]
        line2 = parsed[i + 1]

        if "EOS" in line1 or not line1.strip() or "\t" not in line1:
            continue
        if not line2.strip() or "\t" not in line2:
            continue

        surface1, feature1 = line1.split("\t", 1)
        surface2, feature2 = line2.split("\t", 1)

        pos1 = feature1.split(",")[0]
        pos2 = feature2.split(",")[0]

        if surface1 in NEGATIVE_PREFIX_RULES and pos1 == "接頭辞":
            allowed_pos = NEGATIVE_PREFIX_RULES[surface1]
            if pos2 not in allowed_pos:
                return False  # 不自然な語構成
    return True

def is_valid_pos_set(parsed: list[str]) -> bool:
    for line in parsed:
        if line == "EOS" or not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        pos = cols[4].split("-")[0]
        if pos in {"副詞", "感動詞"}:
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
        and pos != "助詞"
        and is_valid_particle_position(parsed)
        and is_valid_particle_semantics(parsed)
        and is_valid_negative_prefix_usage(parsed)
        and is_valid_inflection(parsed)
        # band is_valid_adverb_placement(parsed)
        and is_valid_pos_set(parsed)
        and is_no_lone_interjection(parsed)
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
    gojuon = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
    return [random.choice(gojuon) for _ in range(n)]

if __name__ == "__main__":
    # words = ["深層岩", "花崗岩", "閃緑岩", "斑レイ岩", "火山岩", "流紋岩", "安山岩", "玄武岩"]
    # words = ["光", "対立", "こう", "サヤ", "エト", "ルナ", "しらべ", "イリス"]
    # words = ["花海咲季", "月村手毬", "藤田ことね", "有村麻央", "姫崎莉波", "十王星南", "雨夜燕"]

    words = random_hira(7)
    print("入力単語(ひらがな):", words)

    hiras = get_first_hiragana_of_words(words)
    print("先頭ひらがな:", hiras)

    pattern = generate_permutations(hiras)
    print("全組み合わせ数:", len(pattern))

    for n in range(3, 5):
        valid = check_counts(pattern, n)
        print(f"単語数 <= {n} の候補数:", len(valid))
        if valid:
            result = most_natural_string(valid)
            print(f"最も自然な文字列: {result}")
            print(tagger.parse(result))
            break