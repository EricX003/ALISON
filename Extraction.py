class Extraction:

    def return_best_pos_n_grams(n, L, text):

        words = word_tokenize(text)
        pos_tokens = pos_tag(words)

        text = [tup[1] for tup in pos_tokens]
        text = ' '.join(text)

        n_grams = ngrams(text, n)

        data = dict(Counter(n_grams))

        list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
        return list_ngrams

    def return_best_word_n_grams(n, L, text):

        all_ngrams =  zip(*[text[i:] for i in range(n)])

        data = dict(Counter(all_ngrams))
        list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
        return list_ngrams

    def return_best_n_grams(n, L, text):
        n_grams = ngrams(text, n)

        data = dict(Counter(n_grams))
        list_ngrams = heapq.nlargest(L, data.keys(), key=lambda k: data[k])
        return list_ngrams

    def find_freq_n_gram_in_txt(text, pos_text, n_gram_lengths, n_grams, pos_n_gram_lengths, pos_n_grams):

        num_ngrams = (sum([len(element) for element in n_grams]) + sum([len(element) for element in pos_n_grams]))
        to_ret = np.zeros([num_ngrams])
        ret_idx = 0

        for idx in range(len(n_grams)):
            num_ngrams = len(Counter(ngrams(text, n_gram_lengths[idx])))

            for n_gram in n_grams[idx]:
                to_ret[ret_idx] = text.count(''.join(n_gram)) / num_ngrams
                ret_idx += 1

        for idx in range(len(pos_n_grams)):
            num_pos_ngrams = len(Counter(ngrams(pos_text, pos_n_gram_lengths[idx])))

            for pos_n_gram in pos_n_grams[idx]:
                to_ret[ret_idx] = pos_text.count(''.join(pos_n_gram)) / num_pos_ngrams
                ret_idx += 1

        return to_ret

    def get_top_idx(freqs, k):
        freqs = [(idx, freqs[idx]) for idx in range(len(freqs))]
        freqs.sort(key = lambda val: val[1], reverse = True)

        return [freqs[idx][0] for idx in range(k)]

    def tag_data(data):
        to_char = lambda x: tags[x] if x in tags else x
        token_and_tag = lambda text: [tup[1] for tup in pos_tag(word_tokenize(text))]
        token_tag_join = lambda text: ''.join([to_char(tag) for tag in token_and_tag(text)])

        data['POS Tagged'] = data[args.text_index].apply(token_tag_join) if 'POS Tagged' not in data.columns else data['POS Tagged']

        return data

    def preprocess_data(data, num_authors, num_total, tag = True):

        print('------------', '\n', 'Tagging...')
        data = tag_data(data) if tag else data

        print('------------', '\n', 'Counting and aggregating texts...')
        number_texts = [0 for idx in range(num_total)]

        texts = ['' for idx in range(num_total)]
        pos_texts = ['' for idx in range(num_total)]

        stop_words = set(stopwords.words('english'))

        for index, row in data.iterrows():
            author = int(row[0])
            number_texts[author] += 1
            filtered_sentence = row[1]
            filtered_pos_sentence = row[2]
            '''
            filtered_sentence = ' '.join([w for w in filtered_sentence.split() if not w in stop_words])
            filtered_sentence = ''.join([w for w in filtered_sentence if w not in set(punctuation)])
            '''
            filtered_sentence = filtered_sentence.strip()
            texts[author] = ' '.join([texts[author], filtered_sentence])
            pos_texts[author] = ''.join([pos_texts[author], filtered_pos_sentence])

        top_idxs = get_top_idx(number_texts, num_authors)

        class_weights = [number_texts[author] for author in top_idxs]

        total  = [texts[idx] for idx in top_idxs]
        total = ' '.join(total)
        pos_total = [pos_texts[idx] for idx in top_idxs]
        pos_total = ''.join(pos_total)

        temp = dict()
        for idx in range(len(top_idxs)):
            temp[top_idxs[idx]] = idx

        top_idxs = temp

        print('------------', '\n', 'Preprocessing complete!')

        return top_idxs, total, pos_total

    def gen_data(data, lengths, n_grams, pos_lengths, pos_n_grams, top_idxs, total, pos_total):

        print('------------', '\n', 'Generating data...')
        X = []
        y = []
        for index, row in data.iterrows():
            if int(row[0]) in top_idxs:
                y.append(top_idxs[int(row[0])])
                X.append(find_freq_n_gram_in_txt(row[1], row[2], lengths, n_grams, pos_lengths, pos_n_grams))

        X = np.array(X)
        y = np.array(y)

        return X, y #sklearn.model_selection.train_test_split(X, y, test_size = 0.15)
