import sys
from random import shuffle
import datetime
import dynet_config
dynet_config.set(autobatch=1)
import dynet as dy
import numpy as np

EPOCHS = 5

LSTM_LAYERS = 1
LSTM_INPUT_DIM = 128
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

def getDataVocab(train_data, input_embedding_enabled, vocab_file, subwords_enabled):
    vocab = {}
    word_count = {}
    size = 0
    if input_embedding_enabled:
        with open(vocab_file) as f:
            for word in f:
                word = word.rstrip()
                vocab[word] = size
                size += 1
    else:
        with open(train_data) as f:
            for line in f:
                if line != '\n':
                    word, tag = line.split()
                    l_word = word.lower()
                    if l_word in word_count:
                        word_count[l_word] += 1
                    else:
                        word_count[l_word] = 1
                    if not vocab.has_key(word):
                        vocab[word] = size
                        size += 1
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
        for w in word_count:
            if word_count[w] <= MIN_WORD_APPEARANCE:
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
    params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, LSTM_INPUT_DIM), init='uniform', scale=(np.sqrt(6)/np.sqrt(250)))


    params["w"] = model.add_parameters((TAGSET_SIZE, 2*LSTM_STATE_DIM))
    params["bias"] = model.add_parameters((TAGSET_SIZE))

    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


def get_model(repr_type, vocab, tag_set):
    if repr_type == 'a':
        return init_model_a(vocab, tag_set)


def get_webms(repr_type, params, line, vocab):
    E = params["lookup"]
    if repr_type == 'a':
        indexes = [getWordIndex(x, vocab) for x in line]
        embs = [E[j] for j in indexes]
        return embs


def train(sen_arr, repr_type, vocab, tag_set, dev_file):
    if dev_file is not None:
        dev_data = load_sentences(dev_file)
    lstms, params, model, trainer = get_model(repr_type, vocab, tag_set)
    f1_lstm, b1_lstm, f2_lstm, b2_lstm = lstms
    w = params["w"]
    b = params["bias"]

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
            webms = get_webms(repr_type, params, sen, vocab)
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

            if i % 100 == 0:
                acc = (good /(good+bad)) * 100
                print("Epoch %d: Train iteration %d: loss=%.4f acc=%%%.2f" % (epoch+1, i, loss_val, acc))

            if (i % 500 == 0) and (dev_file is not None):
                dev_loss, dev_acc = evaluate_dev(dev_data, (lstms, params, model, trainer), vocab)
                print("\t\tEpoch %d: Dev iteration %d: loss=%.4f acc=%%%.2f" % (epoch + 1, i, dev_loss, dev_acc))

    return (lstms, params, model, trainer)


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


def evaluate_dev(dev_data, model_params, vocab):
    (lstms, params, model, trainer) = model_params
    f1_lstm, b1_lstm, f2_lstm, b2_lstm = lstms
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
        webms = get_webms(repr_type, params, sen, vocab)
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


if __name__ == '__main__':
    repr_type = sys.argv[1]
    train_file = sys.argv[2]
    model_file = sys.argv[3]

    dev_file = None
    if "--dev-file" in sys.argv:
        dev_file_option_i = sys.argv.index("--dev-file")
        dev_file = sys.argv[dev_file_option_i+1]

    if repr_type == 'c':
        vocab_file = sys.argv[4]
        embedding_file = sys.argv[5]
    else:
        vocab_file = None

    is_embedding_enabled = (repr_type == 'c')
    vocab = getDataVocab(train_file, is_embedding_enabled, vocab_file, is_embedding_enabled)
    sen_arr = load_sentences(train_file)
    tag_set = load_tag_set(sen_arr)
    revered_tag_set = reverse_tag_set(tag_set)
    model_params = train(sen_arr, repr_type, vocab, tag_set, dev_file)

    print("Ended at " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))




