# -*- coding: utf-8 -*-
# pip install jaconv mecab-python3 unidic-lite
# pip install transformers fugashi ipadic torch
from utils.hiragana_utils import get_first_hiragana_of_words, random_hira, generate_permutations
from utils.bert_scorer import most_natural_string
from utils.pos_validator import check_counts
from utils.word_filters import should_pass
import MeCab

tagger = MeCab.Tagger()
tagger_wakati = MeCab.Tagger("-Owakati")


if __name__ == "__main__":
    words = []
    print("単語を1行ずつ入力")
    while True:
        line = input("> ")
        if line == "":
            break
        words.append(line)

    print("入力された単語リスト:", words)

    # words = random_hira(9)

    print("入力単語(ひらがな):", words)
    hiras = get_first_hiragana_of_words(words)
    print("先頭ひらがな:", hiras)
    pattern = generate_permutations(hiras)
    print("全組み合わせ数:", len(pattern))

    N = 150  # 絞り込み閾値

    for n in range(1, 5):
        valid = check_counts(pattern, n)
        print(f"単語数 <= {n} の候補数:", len(valid))

        if valid:
            if len(valid) > N:
                # 名詞だけで構成されたやつを除外
                valid = should_pass(valid)
                print(f"→ 名詞だけ除外後: {len(valid)} 件")

            if valid:
                result = most_natural_string(valid)
                print(f"最も自然な文字列: {result}")
                print(tagger.parse(result))
                break
            
    # print(tagger.parse("たべられます")) # チェック用