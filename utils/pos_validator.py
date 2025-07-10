import MeCab

tagger = MeCab.Tagger()
tagger_wakati = MeCab.Tagger("-Owakati")

def contains_interjection_or_adverb(parsed: list[str]) -> bool:
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