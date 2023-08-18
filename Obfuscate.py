from Utils import *
from Replacer import *
from NN import *

def clean(texts):
    ret = []

    for elem in texts:

        #elem = elem.upper()
        elem = elem.strip().strip('"').strip("'").replace("  ", "")

        for punct in ["!", ".", "?", ':', ";", ","]:
            elem = elem.replace(f" {punct}", punct)
        elem = elem.replace("' ", "'")
        elem = elem.replace("#", '')

        elem = elem.replace("( ", "(")
        elem = elem.replace(" )", ")")

        ret.append(elem.lower())

    return ret

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
    torch.cuda.empty_cache()

    rep_idx = 0
    for word_idx in range(len(words)):
        if words[word_idx] == "[MASK]":
            words[word_idx] = removePunct(predicted_tokens[rep_idx])

    save.extend(words)
    return save


def main():
    now = datetime.now()

    parser = argparse.ArgumentParser()

    parser.add_argument('--texts', '-t', help = 'Path to texts for obfuscation')
    parser.add_argument('--authors_total', '-at', help='Number of Total Authors in Corpus', default = 10)
    parser.add_argument('--dir', '-f', help = 'Path to the directory containing the trained model')
    parser.add_argument('--model', '-m', help = 'Path to the saved model (model.pt)')
    parser.add_argument('--trial_name', 'tm', help='The Current Trial\'s Name (e.g. Dataset Name)')

    parser.add_argument('--L', '-L', help='L, the number of top POS n-grams to mask', default = 15)
    parser.add_argument('--c', '-c', help='c, the length scaling constant', default = 1.35)
    parser.add_argument('--min_length', '-min', help='The minimum length of POS n-gram to consider for obfuscation', default = 1.35)

    parser.add_argument('--ig_steps', '-ig', help = 'The number of steps for IG importance extraction', default = 1024)

    args = parser.parse_args()

    dir = os.getcwd()
    timestamp = now.strftime("%m.%d.%H.%M.%S")
    save_path = os.path.join(dir, 'Trained_Models', f'{args.trial_name}_{timestamp}')

    os.makedirs(save_path)

    print('------------', '\n', 'Loading Data...')
    with open(args.dir, 'r') as reader:
        lines = [line.partition(' ') for line in reader.readlines()]
        data = pd.DataFrame(data = {
                                    'text' : [line[2] for line in lines],
                                    'label' : [int(line[0]) for line in lines]
                                    })

    features = np.array(pickle.load(open(os.path.join(args.dir, 'features.pkl'), "rb")))
    Scaler= np.array(pickle.load(open(os.path.join(args.dir, 'Scaler.pkl'), "rb")))
    num_char = features[0].size
    num_pos = features[1].size
    features = features.flatten().tolist()

    ngram_reps = []
    for idx, row in data.iterrows():
        ngram_reps.append(ngram_rep(row[0], row[1], features))
    ngram_reps = Scaler.fit_transform(np.array(ngram_reps))

    ig_set = torch.utils.data.DataLoader(Loader(ngram_reps, data['label']), batch_size=1, shuffle=False)

    model = Model(len(os.path.join(args.dir, 'X_test.pkl')[0]), args.authors_total)
    model.load_state_dict(torch.load(os.path.join(args.dir, 'model.pt')))
    model.eval()

    ig = IntegratedGradients(model)

    all = []
    torch.cuda.empty_cache()
    for data, label in ig_set:

        attributions = ig.attribute(data.cuda(), target = label.to(torch.int64).cuda(), n_steps = args.ig_steps)
        attributions = attributions.tolist()

        for attribution in attributions:
            all.append(attribution)

        torch.cuda.empty_cache()
        del attributions

    data['attribution'] = all

    rev_tags = dict()
    for tag in tags:
        rev_tags[tags[tag]] = tag

    isValid = lambda index : index >= index >= num_char and index < num_char + num_pos
    to_char = lambda x: tags[x] if x in tags else x
    token_and_tag = lambda text: [tup[1] for tup in pos_tag(tokenize(text))]
    token_tag_join = lambda text: ''.join([to_char(tag) for tag in token_and_tag(text)])

    for idx, row in data.iterrows():

        torch.cuda.empty_cache()

        text = row['text']
        attribution = row['attribution']

        mult = [args.c ** len(feature) for feature in features]
        attribution = np.multiply(attribution, mult)

        ranked_indexes = np.argsort(np.array(ranked_indexes))
        ranked_indexes = [elem for elem in ranked_indexes if isValid(elem)]
        ranked_indexes.reverse()
        to_replace = [features[elem] for elem in ranked_indexes]

        to_replace = [replace for replace in to_replace if len(replace) > args.min_length]
        to_replace = to_replace[ : args.L]

        words = tokenize(text)

        retagged = pos_tag(words)
        retagged = [to_compressed(tup[1]) for tup in retagged]

        intervals = []

        for replace in to_replace:

            starts = [i for i in range(len(retagged) - len(replace)) if replace == "".join(retagged[i:i + len(replace)])]

            for start in starts:
                intervals.append([start, start + len(replace)])

        changed = [False] * len(words)

        for interval in intervals:
            if not any(changed[interval[0] : interval[1]]):
                words = replace_interval(words, interval)
                changed[interval[0] : interval[1]] = [True] * (interval[1] - interval[0])

        obfuscated_texts.append(" ".join(words))

    obfuscated_texts = clean(obfuscated_texts)
    with open(os.path.join(save_path, 'adversarial_texts.txt'), 'w') as writer:
        writer.writelines(obfuscated_texts)

if __name__ == "__main__":
    main()
