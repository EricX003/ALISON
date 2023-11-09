from transformers import BertTokenizer, BertForMaskedLM
import torch
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
replacer_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
replacer = BertForMaskedLM.from_pretrained("bert-large-uncased").to(device)


def replace_interval(words, interval):

    words[interval[0] : interval[1]] =  ["[MASK]"] * (interval[1] - interval[0])
    save = []

    if interval[1] > 512:
        split = interval[1] - 512
        save = words[ :split]
        words = words[split: ]

    inputs = replacer_tokenizer(" ".join(words), return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    token_ids = replacer_tokenizer.encode(" ".join(words), return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    masked_positions = [idx for idx in range(len(words)) if words[idx] == "[MASK]"]

    outputs = replacer(**inputs)

    predictions = outputs[0]
    sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)

    predicted_index = [torch.argmax(predictions[0, i]).item() for i in range(len(predictions[0]))]
    predicted_token = [replacer_tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(len(predictions[0]))]
    predicted_tokens = [predicted_token[pos] for pos in masked_positions]

    del inputs, token_ids, outputs, predictions, sorted_idx, sorted_preds

    rep_idx = 0
    for word_idx in range(len(words)):
        if words[word_idx] == "[MASK]":
            words[word_idx] = re.sub(r'[^\w\s]', '', predicted_tokens[rep_idx])

    save.extend(words)
    return save
