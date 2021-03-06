import sys
from random import shuffle
import datetime
import dynet_config
dynet_config.set(autobatch=1)
import dynet as dy
import numpy as np
import matplotlib.pyplot as plt
import pickle

EPOCHS = 5

LSTM_LAYERS = 1
LSTM_INPUT_DIM = 50
LSTM_STATE_DIM = 50


MIN_WORD_APPEARANCE = 2
uncommon_words = set()
SUFF_UUUNKKK = 'SUFF_UUUNKKK'
PRE_UUUNKKK = 'PRE_UUUNKKK'
UUUNKKK = 'UUUNKKK'
UNK_NUM = 'UNK_num'
UNK_ALLCAP = 'UNK_ALLCAP'
UNK_CAP_START = 'UNK_CapStart'


def load_sentences(data):
    """
    Converting file to array of sentences
    :param data:
    :return: sen_arr
    """
    f = open(data)
    sen_arr = []
    sen = []
    for line in f:
        if line != '\n':
            sen.append(line)
        else:
            sen_arr.append(sen)
            sen = []
    if len(sen) > 0:
        sen_arr.append(sen)
    f.close()

    return sen_arr


def getDataVocab(train_data, input_embedding_enabled, vocab_file, subwords_enabled, char_embedding, word_and_char_emb):
    vocab = {}
    key_count = {}  # == word_count  == char_count
    size = 0
    my_emb = True
    if char_embedding or input_embedding_enabled:
        my_emb = False
    if word_and_char_emb:
        char_embedding = True

    # already trained embedding
    if input_embedding_enabled:
        with open(vocab_file) as f:
            for word in f:
                word = word.rstrip()
                vocab[word] = size
                size += 1

    elif char_embedding:
        with open(train_data) as f:
            for line in f:
                if line != '\n':
                    word, tag = line.split()
                    # l_word = word.lower()
                    for char in word:
                        if char in key_count:
                            key_count[char] += 1
                        else:
                            key_count[char] = 1
                        if not vocab.has_key(char):
                            vocab[char] = size
                            size += 1
        vocab[UUUNKKK] = size
        size += 1

    # my embedding
    if my_emb:
        with open(train_data) as f:
            for line in f:
                if line != '\n':
                    word, tag = line.split()
                    l_word = word.lower()
                    if l_word in key_count:
                        key_count[l_word] += 1
                    else:
                        key_count[l_word] = 1
                    if not vocab.has_key(word):
                        vocab[word] = size
                        size += 1
        if not vocab.has_key(UUUNKKK):
            vocab[UUUNKKK] = size
            size += 1

    if subwords_enabled:
        subwords_dict = {}
        for key in vocab:
            if len(key) > 2:
                pre_key = key[:3]
                pre_key = "PRE_" + pre_key
                suff_key = key[-3:]
                suff_key = "SUFF_" + suff_key
                if not subwords_dict.has_key(pre_key):
                    subwords_dict[pre_key] = 0
                if not subwords_dict.has_key(suff_key):
                    subwords_dict[suff_key] = 0

        for key, val in subwords_dict.iteritems():
            if not vocab.has_key(key):
                vocab[key] = size
                size += 1

        vocab[PRE_UUUNKKK] = size
        size += 1
        vocab[SUFF_UUUNKKK] = size
        size += 1

    if my_emb or (not char_embedding): # if not char embedding, if any word embedding
        vocab[UNK_CAP_START] = size
        size += 1
        vocab[UNK_ALLCAP] = size
        size += 1
        vocab[UNK_NUM] = size
        size += 1

    global uncommon_words
    if input_embedding_enabled:
        uncommon_words = set()
    else:
        for w in key_count:
            if key_count[w] <= MIN_WORD_APPEARANCE:
                uncommon_words.add(w)

    return vocab


def load_tag_set(sen_arr):
    """
    Getting the possible labels of the tagset of this train data/
    :param train_data:
    :return:
    """

    tag_set = {}
    num = 0
    for sentence in sen_arr:
        for line in sentence:
            if line != '\n':
                word, tag = line.split()
                if tag not in tag_set:
                    tag_set[tag] = num
                    num += 1
    return tag_set


def reverse_tag_set(tag_set):
    tag_set_rev = {}
    for tag in tag_set:
        tag_set_rev[tag_set[tag]] = tag

    return tag_set_rev


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def init_model_a(vocab, tag_set):
    TAGSET_SIZE = len(tag_set)
    VOCAB_SIZE = len(vocab)

    model = dy.ParameterCollection()
    f1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    b1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    f2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2*LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    b2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2*LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    lstms = (f1_lstm, b1_lstm, f2_lstm, b2_lstm)

    params = {}
    params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, LSTM_INPUT_DIM), init='uniform', scale=(np.sqrt(6)/np.sqrt(LSTM_INPUT_DIM)))


    params["w"] = model.add_parameters((TAGSET_SIZE, 2*LSTM_STATE_DIM))
    params["bias"] = model.add_parameters((TAGSET_SIZE))

    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


def init_model_b(vocab, tag_set):
    TAGSET_SIZE = len(tag_set)
    VOCAB_SIZE = len(vocab)

    model = dy.ParameterCollection()

    c_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model)
    f1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    b1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    f2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2 * LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    b2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2 * LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    lstms = (c_lstm, f1_lstm, b1_lstm, f2_lstm, b2_lstm)

    params = {}
    params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, LSTM_INPUT_DIM), init='uniform',
                                                   scale=(np.sqrt(6) / np.sqrt(LSTM_INPUT_DIM)))

    params["w"] = model.add_parameters((TAGSET_SIZE, 2 * LSTM_STATE_DIM))
    params["bias"] = model.add_parameters((TAGSET_SIZE))

    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


def init_model_c(vocab, tag_set, embedding_file):
    model = dy.ParameterCollection()
    params = {}
    TAGSET_SIZE = len(tag_set)
    VOCAB_SIZE = len(vocab)

    with open(embedding_file) as f:
        numbers = 0
        print "load word vectors ..."
        input_wordVectors = []
        for line in f:
            number_strings = line.split()  # Split the line on runs of whitespace
            numbers = [float(n) for n in number_strings]  # Convert to floats
            input_wordVectors.append(numbers)
        while len(input_wordVectors) < len(vocab):
            eps = np.sqrt(6) / np.sqrt(len(numbers))
            vec = np.random.uniform(-eps, eps, len(numbers))
            input_wordVectors.append(vec)
    params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, len(input_wordVectors[0])))
    params["lookup"].init_from_array(np.array(input_wordVectors))

    f1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    b1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    f2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2 * LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    b2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2 * LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    lstms = (f1_lstm, b1_lstm, f2_lstm, b2_lstm)

    params["w"] = model.add_parameters((TAGSET_SIZE, 2 * LSTM_STATE_DIM))
    params["bias"] = model.add_parameters((TAGSET_SIZE))

    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


def init_model_d(vocab, tag_set):
    TAGSET_SIZE = len(tag_set)
    VOCAB_SIZE = len(vocab)
    params = {}

    model = dy.ParameterCollection()

    c_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model)
    f1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    b1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    f2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2 * LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    b2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2 * LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    lstms = (c_lstm, f1_lstm, b1_lstm, f2_lstm, b2_lstm)

    params["w_con"] = model.add_parameters((LSTM_INPUT_DIM, 2*LSTM_INPUT_DIM))
    params["b_con"] = model.add_parameters((LSTM_INPUT_DIM))

    params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, LSTM_INPUT_DIM), init='uniform',
                                                   scale=(np.sqrt(6) / np.sqrt(LSTM_INPUT_DIM)))
    params["w"] = model.add_parameters((TAGSET_SIZE, 2 * LSTM_STATE_DIM))
    params["bias"] = model.add_parameters((TAGSET_SIZE))

    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


def get_model(repr_type, vocab, tag_set, embedding_file):
    if repr_type == 'a':
        return init_model_a(vocab, tag_set)

    elif repr_type == 'b':
        return init_model_b(vocab, tag_set)

    elif repr_type == 'c':
        return init_model_c(vocab, tag_set, embedding_file)

    elif repr_type == 'd':
        return init_model_d(vocab, tag_set)


def get_webms(repr_type, params, line, vocab, char_lstm):
    E = params["lookup"]

    if repr_type == 'a':
        indexes = [getWordIndex(x, vocab) for x in line]
        embs = [E[j] for j in indexes]

    elif repr_type == 'c':
        indexes = [getWordIndex(x, vocab) for x in line]
        just_words_embs = [E[j] for j in indexes]
        preWordIndexVector, suffWordIndexVector = getVectorPreSuffWordIndexes(line, vocab)
        pre_embs = [E[j] for j in preWordIndexVector]
        suff_embs = [E[j] for j in suffWordIndexVector]
        embs = [dy.esum([just_words_embs[i], pre_embs[i], suff_embs[i]]) for i,some_val in enumerate(just_words_embs)]

    elif repr_type == 'b':
        sen_embs = []
        for word_and_tag in line:
            s = char_lstm.initial_state()
            word = word_and_tag.split()[0]
            c_word = [(c) for c in word]
            c_indexes = [getCharIndex(x, vocab) for x in c_word]
            c_embs = [E[j] for j in c_indexes]
            for c_e in c_embs:
                s = s.add_input(c_e)

            sen_embs.append(s.output())
        embs = sen_embs

    elif repr_type == 'd':
        w = params['w_con']
        b = params['b_con']

        # word emb
        indexes = [getWordIndex(x, vocab) for x in line]
        words_embs = [E[j] for j in indexes]

        # char emb
        sen_embs = []
        for word_and_tag in line:
            s = char_lstm.initial_state()
            word = word_and_tag.split()[0]
            c_word = [(c) for c in word]
            c_indexes = [getCharIndex(x, vocab) for x in c_word]
            c_embs = [E[j] for j in c_indexes]
            for c_e in c_embs:
                s = s.add_input(c_e)

            sen_embs.append(s.output())
        char_embs = sen_embs

        con_embs = [dy.concatenate([words_embs[x], char_embs[x]]) for x in range(len(words_embs))]

        embs = [((w*e)+b) for e in con_embs]

    return embs


def getVectorPreSuffWordIndexes(sen, vocab):
    pre_all = []
    suff_all = []

    for word in sen:
        pre, suff = getPreSuffWordIndex(word, vocab)
        pre_all.append(pre)
        suff_all.append(suff)

    return pre_all, suff_all


def getPreSuffWordIndex(word, vocab, length=3):
    word = word.split()[0]
    pre_word = word[:length]
    pre_word = "PRE_" + pre_word
    suff_word = word[-length:]
    suff_word = "SUFF_" + suff_word
    if vocab.has_key(pre_word):
        pre_val = vocab[pre_word]
    else:
        pre_val = vocab['PRE_UUUNKKK']
    if vocab.has_key(suff_word):
        suff_val = vocab[suff_word]
    else:
        suff_val = vocab['SUFF_UUUNKKK']

    return pre_val, suff_val


def train(sen_arr, repr_type, vocab, tag_set, dev_file, embedding_file):
    if dev_file is not None:
        dev_data = load_sentences(dev_file)
    lstms, params, model, trainer = get_model(repr_type, vocab, tag_set, embedding_file)
    if repr_type != 'b' and repr_type != 'd':  # repr b has 5 lstms (clstm)
        f1_lstm, b1_lstm, f2_lstm, b2_lstm = lstms
        char_lstm = None
    else:
        char_lstm, f1_lstm, b1_lstm, f2_lstm, b2_lstm = lstms
    w = params["w"]
    b = params["bias"]

    dev_acc_arr = []

    for epoch in range(EPOCHS):
        shuffle(sen_arr)
        print("Epoch " + str(epoch+1) + " time started at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        good = bad = 0.0
        for i in range(len(sen_arr)):
            sen = sen_arr[i]
            dy.renew_cg()
            f1_lstm_init = f1_lstm.initial_state()
            b1_lstm_init = b1_lstm.initial_state()
            f2_lstm_init = f2_lstm.initial_state()
            b2_lstm_init = b2_lstm.initial_state()
            webms = get_webms(repr_type, params, sen, vocab, char_lstm)

            fw1_exps = f1_lstm_init.transduce(webms)
            bw1_exps = b1_lstm_init.transduce(reversed(webms))
            bi1 = [dy.concatenate([f,b_exp]) for f,b_exp in zip(fw1_exps, reversed(bw1_exps))]

            fw2_exps = f2_lstm_init.transduce(bi1)
            bw2_exps = b2_lstm_init.transduce(reversed(bi1))
            bi2 = [dy.concatenate([f,b_exp]) for f,b_exp in zip(fw2_exps, reversed(bw2_exps))]

            probs_sequence = [dy.softmax((w*out)+b) for out in bi2]
            yhat_sequence = [np.argmax(probs.value()) for probs in probs_sequence]
            golds_sequence = [get_tag_i(sen, i_gold) for i_gold in range(len(sen))]

            for yhat, gold in zip(yhat_sequence, golds_sequence):
                if yhat == gold:
                    if gold != tag_set['O']:
                        good += 1
                else:
                    bad += 1

            loss_sequence = []
            for i_prob, probs in enumerate(probs_sequence):
                tag = get_tag_i(sen, i_prob)
                loss_sequence.append(-(dy.log(dy.pick(probs, tag))))
            loss = dy.esum(loss_sequence)
            loss_val = loss.value()

            loss.backward()
            trainer.update()

            if i % 100 == 0 and (i > 0):
                acc = (good /(good+bad)) * 100
                print("Epoch %d: Train iteration %d: loss=%.4f acc=%%%.2f" % (epoch+1, i, loss_val, acc))

            if (i % 500 == 0) and (dev_file is not None) and (i > 0):
                dev_loss, dev_acc = evaluate_dev(repr_type, dev_data, (lstms, params, model, trainer), vocab)
                print("=======Dev iteration: loss=%.4f acc=%%%.2f=======" % (dev_loss, dev_acc))
                dev_acc_arr.append(dev_acc)

    return (lstms, params, model, trainer), dev_acc_arr


def get_tag_i(line, i):
    return int(tag_set[line[i].split()[1]])


def getWordIndex(word_and_tag, vocab):
    """
    get word's index from the vocab
    :param sen, k
    :return:
    """
    word = word_and_tag.split()[0]
    if vocab.has_key(word) and word.lower() not in uncommon_words:
        i = vocab[word]
    elif vocab.has_key(word.lower()) and word.lower() not in uncommon_words:
        i = vocab[word.lower()]
    elif word.isupper():
        i = vocab[UNK_ALLCAP]
    elif word[0].isupper():
        i = vocab[UNK_CAP_START]
    elif is_number(word):
        i = vocab[UNK_NUM]
    else:
        i = vocab[UUUNKKK]

    return i


def getCharIndex(char_str, vocab):
    """
    get chars's index from the vocab
    :param sen, k
    :return:
    """
    if vocab.has_key(char_str):
        i = vocab[char_str]
    elif is_number(char_str):
        i = vocab[UNK_NUM]
    else:
        i = vocab[UUUNKKK]

    return i


def evaluate_dev(repr_type, dev_data, model_params, vocab):
    (lstms, params, model, trainer) = model_params
    if repr_type != 'b' and repr_type != 'd':  # repr b has 5 lstms (clstm)
        f1_lstm, b1_lstm, f2_lstm, b2_lstm = lstms
        char_lstm = None
    else:
        char_lstm, f1_lstm, b1_lstm, f2_lstm, b2_lstm = lstms
    w = params["w"]
    b = params["bias"]
    good = bad = 0.0
    counter = 0
    total_loss = 0

    for i in range(len(dev_data)):
        sen = dev_data[i]
        dy.renew_cg()
        f1_lstm_init = f1_lstm.initial_state()
        b1_lstm_init = b1_lstm.initial_state()
        f2_lstm_init = f2_lstm.initial_state()
        b2_lstm_init = b2_lstm.initial_state()
        webms = get_webms(repr_type, params, sen, vocab, char_lstm)
        fw1_exps = f1_lstm_init.transduce(webms)
        bw1_exps = b1_lstm_init.transduce(reversed(webms))
        bi1 = [dy.concatenate([f, b_exp]) for f, b_exp in zip(fw1_exps, reversed(bw1_exps))]

        fw2_exps = f2_lstm_init.transduce(bi1)
        bw2_exps = b2_lstm_init.transduce(reversed(bi1))
        bi2 = [dy.concatenate([f, b_exp]) for f, b_exp in zip(fw2_exps, reversed(bw2_exps))]
        probs_sequence = [dy.softmax((w * out) + b) for out in bi2]
        yhat_sequence = [np.argmax(probs.value()) for probs in probs_sequence]
        golds_sequence = [get_tag_i(sen, i_gold) for i_gold in range(len(sen))]
        for yhat, gold in zip(yhat_sequence, golds_sequence):
            counter += 1
            if yhat == gold:
                if gold != tag_set['O']:
                    good += 1
            else:
                bad += 1

        loss_sequence = []
        for i_prob, probs in enumerate(probs_sequence):
            tag = get_tag_i(sen, i_prob)
            loss_sequence.append(-(dy.log(dy.pick(probs, tag))))
        loss = dy.esum(loss_sequence)
        total_loss += loss.value()
    total_loss /= counter
    acc = (good / (good + bad)) * 100

    return total_loss, acc


def analyse_statistics(dev_acc_arr, repr_type):
    np_file_name = 'stat/' + repr_type + '.txt'
    with open(np_file_name, 'w') as f:
        np.savetxt(f, dev_acc_arr)

    plt_file_name = 'stat/' + repr_type + '.png'
    plt.plot(dev_acc_arr)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    #plt.xticks(np.arange(0, step=5))
    plt.title(repr_type)
    plt.legend()
    plt.savefig(plt_file_name)
    plt.close()


def saveModel(model_file, model_params):
    lstms, params, model, trainer = model_params
    model.save(model_file)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    repr_type = sys.argv[1]
    train_file = sys.argv[2]
    model_file = sys.argv[3]

    dev_file = None
    if "--dev-file" in sys.argv:
        dev_file_option_i = sys.argv.index("--dev-file")
        dev_file = sys.argv[dev_file_option_i+1]

    # not repr c
    vocab_file = None
    embedding_file = None
    if_c = False

    # not repr b
    if_b = False

    # not repr d
    word_and_char_emb = False

    if repr_type == 'c':
        if "--vocab" in sys.argv:
            vocab_file_option_i = sys.argv.index("--vocab")
            vocab_file = sys.argv[vocab_file_option_i + 1]
        else:
            vocab_file = 'data/vocab.txt'
        if "--wordVectors" in sys.argv:
            wordVectors_file_option_i = sys.argv.index("--wordVectors")
            embedding_file = sys.argv[wordVectors_file_option_i + 1]
        else:
            embedding_file = 'data/wordVectors.txt'
        if_c = True

    elif repr_type == 'b':
        if_b = True
    elif repr_type == 'd':
        word_and_char_emb = True

    vocab = getDataVocab(train_file, if_c, vocab_file, if_c, if_b, word_and_char_emb)
    sen_arr = load_sentences(train_file)
    tag_set = load_tag_set(sen_arr)
    revered_tag_set = reverse_tag_set(tag_set)
    model_params, dev_acc_arr = train(sen_arr, repr_type, vocab, tag_set, dev_file, embedding_file)
    saveModel(model_file, model_params)
    vocab_file_name = repr_type + '_vocab'
    save_obj(vocab, vocab_file_name)
    save_obj(revered_tag_set, 'tag_set')

    if ("--analyse" in sys.argv) and (dev_file is not None):
        analyse_statistics(dev_acc_arr, repr_type)


    print("Ended at " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

