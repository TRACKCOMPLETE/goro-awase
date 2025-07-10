from transformers import BertJapaneseTokenizer, BertForMaskedLM
import torch

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