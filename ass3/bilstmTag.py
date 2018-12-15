import sys
import dynet_config
dynet_config.set(autobatch=1)
import dynet as dy
import numpy as np
import pickle


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


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_model(repr_type, vocab, tag_set, trained_model):
    if repr_type == 'a':
        return init_model_a(vocab, tag_set, trained_model)

    elif repr_type == 'b':
        return init_model_b(vocab, tag_set, trained_model)

    elif repr_type == 'c':
        return init_model_c(vocab, tag_set, trained_model)

    elif repr_type == 'd':
        return init_model_d(vocab, tag_set, trained_model)


def init_model_a(vocab, tag_set, trained_model):
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

    model.populate(trained_model)
    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


def init_model_b(vocab, tag_set, trained_model):
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

    model.populate(trained_model)
    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


def init_model_c(vocab, tag_set, trained_model):
    model = dy.ParameterCollection()
    params = {}
    TAGSET_SIZE = len(tag_set)
    VOCAB_SIZE = len(vocab)

    params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, LSTM_INPUT_DIM), init='uniform', scale=(np.sqrt(6)/np.sqrt(LSTM_INPUT_DIM)))

    f1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    b1_lstm = dy.LSTMBuilder(LSTM_LAYERS, LSTM_INPUT_DIM, LSTM_STATE_DIM, model)
    f2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2 * LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    b2_lstm = dy.LSTMBuilder(LSTM_LAYERS, 2 * LSTM_STATE_DIM, LSTM_STATE_DIM, model)
    lstms = (f1_lstm, b1_lstm, f2_lstm, b2_lstm)

    params["w"] = model.add_parameters((TAGSET_SIZE, 2 * LSTM_STATE_DIM))
    params["bias"] = model.add_parameters((TAGSET_SIZE))

    model.populate(trained_model)
    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


def init_model_d(vocab, tag_set, trained_model):
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

    model.populate(trained_model)
    trainer = dy.AdamTrainer(model)
    return (lstms, params, model, trainer)


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


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def evaluateTest(repr_type, test_data, trained_model, vocab):
    lstms, params, model, trainer = get_model(repr_type, vocab, tag_set, trained_model)

    if repr_type != 'b' and repr_type != 'd':  # repr b has 5 lstms (clstm)
        f1_lstm, b1_lstm, f2_lstm, b2_lstm = lstms
        char_lstm = None
    else:
        char_lstm, f1_lstm, b1_lstm, f2_lstm, b2_lstm = lstms
    w = params["w"]
    b = params["bias"]

    predictions = []

    for i in range(len(test_data)):
        sen = test_data[i]
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

        [predictions.append([sen[w_i], yhat_sequence[w_i]]) for w_i,just_any in enumerate(sen)]
        predictions.append([])

    return predictions


def write2file(preds, tag_set, output_file):
    with open(output_file, 'w') as f:
        for pair in preds:
            if pair:
                word, tag_i = pair
                word = word.split()[0]
                tag = tag_set[tag_i]
                string = word + " " + tag + '\n'
                f.write(string)
            else:
                f.write('\n')



if __name__ == '__main__':
    repr_type = sys.argv[1]
    model_file = sys.argv[2]
    input_file = sys.argv[3]

    if "--output" in sys.argv:
        output_file_option_i = sys.argv.index("--output")
        output_file = sys.argv[output_file_option_i + 1]
    else:
        output_file = "test.pred"

    sen_arr = load_sentences(input_file)
    vocab_file_name = repr_type + '_vocab'
    vocab = load_obj(vocab_file_name)
    tag_set = load_obj('tag_set')
    preds = evaluateTest(repr_type, sen_arr, model_file, vocab)

    write2file(preds, tag_set, output_file)




