from Utils import *
from NN import *

def main():
    now = datetime.now()

    gc.collect()

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', '-T', help = 'Path to Training Data', default='../Data/train.txt')
    parser.add_argument('--authors_total', '-at', help='Number of Total Authors in Corpus', default = 10)

    parser.add_argument('--trial_name', 'tm', help='The Current Trial\'s Name (e.g. Dataset Name)')
    parser.add_argument('--test_size', 'ts', help = 'Proportion of data to use for testing', default=0.15)

    parser.add_argument('--top_ngrams', '-tng', help='t, The Number of Top Character and POS-ngrams to Retain', default = 256)
    parser.add_argument('--V', '-V', help='V, the set of n-gram lengths to use', default = [1, 2, 3, 4])

    parser.add_argument('--batch_size', '-bs', help = 'Batch Size', default = 512)
    parser.add_argument('--learning_rate', '-lr', help = 'Learning Rate', default = 0.01)
    parser.add_argument('--epochs', '-e', help='Number of Training Epochs', default = 30)
    parser.add_argument('--weight_decay', '-wd', help='Weight Decay Constant', default = 0.00)
    parser.add_argument('--momentum', '-m', help='Momentum Constant', default = 0.90)
    parser.add_argument('--step', '-s', help='Scheduler Step Size', default = 3)
    parser.add_argument('--gamma', '-g', help='Scheduler Gamma Constant', default = 0.30)

    args = parser.parse_args()

    dir = os.getcwd()
    timestamp = now.strftime("%m.%d.%H.%M.%S")
    trial_name = f'{args.trial_name}_{timestamp}'
    save_path = os.path.join(dir, 'Trained_Models', trial_name)

    os.makedirs(save_path)

    with open(parser.train, 'r') as reader:
        lines = [line.partition(' ') for line in reader.readlines()]
        labels = [int(line[0]) for line in lines]
        texts = [line[2] for line in lines]

        data = pd.DataFrame(data = {'label' : labels, 'text' : texts})

    print('------------', '\n', 'Tagging...')
    data['POS'] = tag(data['text'])

    print('------------', '\n', 'Counting and aggregating texts...')
    number_texts = [0 for idx in range(args.authors_total)]

    texts = ['' for idx in range(args.authors_total)]
    pos_texts = ['' for idx in range(args.authors_total)]

    total = ' '.join(texts)
    pos_total = ''.join(pos_texts)

    for index, row in data.iterrows():
        author = int(row[0])
        number_texts[author] += 1
        filtered_sentence = row['text'].replace('\n', '').strip()

        texts[author] = ' '.join([texts[author], filtered_sentence])
        pos_texts[author] = ''.join([pos_texts[author], row[2]])

    print('------------', '\n', 'Preprocessing complete!')
    print('------------', '\n', 'Generating Char n-grams...')

    n_grams = [return_best_n_grams(n, args.top_ngrams, total) for n in [1, 2, 3, 4]]

    print('------------', '\n', 'Generating POS n-grams...')

    pos_n_grams = [return_best_n_grams(n, args.top_ngrams, pos_total) for n in [1, 2, 3, 4]]

    print('------------', '\n', 'Generating Word n-grams...')

    word_n_grams = [return_best_word_n_grams(n, args.top_ngrams, tokenize(total)) for n in [1, 2, 3, 4]]

    print('------------', '\n', 'Generating data...')
    X = []
    y = []
    processed = 0
    for index, row in data.iterrows():
        if(processed % 1000 == 0):
            print(f'{processed} texts processed')

        y.append(int(row['label']))
        X.append(ngram_rep(row['text'], row['POS_text'], args.V, n_grams, args.V, pos_n_grams, args.V, word_n_grams))

        processed += 1

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=args.test_size, random_state=1, shuffle=False, stratify=y)

    print('------------', '\n', 'Scaling, Loading, and Shuffling Data')
    Scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train = Scaler.transform(X_train)
    X_test = Scaler.transform(X_test)

    training_Loader = Loader(X_train, y_train)
    validation_Loader = Loader(X_test, y_test)

    training_set = torch.utils.data.DataLoader(training_Loader, batch_size = args.batch_size, shuffle = True)
    validation_set = torch.utils.data.DataLoader(validation_Loader, batch_size = args.batch_size, shuffle = False)

    pickle.dump(X_train, open(os.path.join(save_path, 'X_train.pkl'), 'wb'))
    pickle.dump(y_train, open(os.path.join(save_path, 'y_train.pkl'), 'wb'))
    pickle.dump(X_test, open(os.path.join(save_path, 'X_test.pkl'), 'wb'))
    pickle.dump(y_test, open(os.path.join(save_path, 'y_test.pkl'), 'wb'))

    features = [n_grams, pos_n_grams, word_n_grams]
    pickle.dump(features, open(os.path.join(save_path, 'features.pkl'), 'wb'))

    model = Model(len(X_train[0]), args.num_authors)
    #model.load_state_dict(torch.load('/content/gdrive/MyDrive/PSU_REU/Models/TuringBench/POS_NN/Blind_Black/model.pt'))

    loss_function = nn.CrossEntropyLoss(weight = torch.Tensor(number_texts).to(device))
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.step, gamma = args.gamma)

    model = train_and_eval(model, training_set, validation_set, loss_function, optimizer, scheduler, save_path=save_path, EPOCHS = args.epochs, save_epoch=10)

    #os.makedirs('/content/gdrive/MyDrive/PSU_REU/Models/BLOG/POS_NN/BLIND/Skip')
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))

    print('------------', '\n', 'Training Done!')

if __name__ == "__main__":
    main()
