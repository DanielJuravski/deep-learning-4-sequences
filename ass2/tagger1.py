import dynet as dy
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime


N1 = 50
EPOCHS = 5

LR = 0.0002
REGULARIZATION_FACTOR = 0

BATCH_SIZE = 1000


IGNORE_O = True
MIN_WORD_APPEARANCE = 5
TRAIN_CUTOFF = float("inf")

uncommon_words = set()

train_losses = []
train_acc = []



UUUNKKK = 'UUUNKKK'
UNK_NUM = 'UNK_num'
UNK_ALLCAP = 'UNK_ALLCAP'
UNK_CAP_START = 'UNK_CapStart'

def getDataVocab(train_data, if_input_embedding, vocab_file, if_subwords):
    vocab = {}
    word_count = {}
    size = 0
    if if_input_embedding == 'embedding':
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

    if if_subwords == 'subwords':
        subwords_dict = {}
        for key in vocab:
            if len(key) > 2: #and key.isalpha()
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

        vocab['PRE_UUUNKKK'] = size
        size += 1
        vocab['SUFF_UUUNKKK'] = size
        size += 1

    vocab['/S/S'] = size
    size += 1
    vocab['/S'] = size
    size += 1
    vocab['/E/E'] = size
    size += 1
    vocab['/E'] = size
    size += 1
    vocab[UNK_CAP_START] = size
    size += 1
    vocab[UNK_ALLCAP] = size
    size += 1
    vocab[UNK_NUM] = size
    size += 1

    global uncommon_words
    if if_input_embedding == 'embedding':
        uncommon_words = set()
    else:
        for w in word_count:
            if word_count[w] <= MIN_WORD_APPEARANCE:
                uncommon_words.add(w)

    return vocab


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


def load_tag_set(train_data):
    """
    Getting the possible labels of the tagset of this train data/
    :param train_data:
    :return:
    """
    train_f = open(train_data)
    tag_set = {}
    num = 0
    for line in train_f:
        if line != '\n':
            word, tag = line.split()
            if tag not in tag_set:
                tag_set[tag] = num
                num += 1

    train_f.close()

    return tag_set


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
    sen_arr.append(sen)
    f.close()

    return sen_arr


def getWordIndex(sen, k, vocab):
    """
    get word's index from the vocab
    :param sen, k
    :return:
    """
    word = sen[k].split()[0]
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


def getVectorWordIndexes(i, sen, vocab):
    sen_len = len(sen)

    if i < 2:
        wpp = vocab['/S/S']
    else:
        wpp = getWordIndex(sen, i-2, vocab)

    if i < 1:
        wp = vocab['/S']
    else:
        wp = getWordIndex(sen, i-1, vocab)

    wi = getWordIndex(sen, i, vocab)

    if i > sen_len - 2:
        wn = vocab['/E']
    else:
        wn = getWordIndex(sen, i+1, vocab)

    if i > sen_len - 3:
        wnn = vocab['/E/E']
    else:
        wnn = getWordIndex(sen, i+2, vocab)

    total_i = [wpp, wp, wi, wn, wnn]

    return total_i


def getPreSuffWordIndex(sen, j, vocab):
    word = sen[j]
    pre_word = word[:3]
    pre_word = "PRE_" + pre_word
    suff_word = word[-3:]
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


def getVectorPreSuffWordIndexes(i, sen, vocab):
    sen_len = len(sen)

    if i < 2:
        wpp_pre = vocab['/S/S']
        wpp_suff = vocab['/S/S']
    else:
        wpp_pre, wpp_suff = getPreSuffWordIndex(sen, i - 2, vocab)

    if i < 1:
        wp_pre = vocab['/S']
        wp_suff = vocab['/S']
    else:
        wp_pre, wp_suff = getPreSuffWordIndex(sen, i - 1, vocab)

    wi_pre, wi_suff = getPreSuffWordIndex(sen, i, vocab)

    if i > sen_len - 2:
        wn_pre = vocab['/E']
        wn_suff = vocab['/E']
    else:
        wn_pre, wn_suff = getPreSuffWordIndex(sen, i + 1, vocab)

    if i > sen_len - 3:
        wnn_pre = vocab['/E/E']
        wnn_suff = vocab['/E/E']
    else:
        wnn_pre, wnn_suff = getPreSuffWordIndex(sen, i + 2, vocab)

    pre_total_i = [wpp_pre, wp_pre, wi_pre, wn_pre, wnn_pre]
    suff_total_i = [wpp_suff, wp_suff, wi_suff, wn_suff, wnn_suff]

    return pre_total_i, suff_total_i


def train_model(sen_arr, vocab, tag_set, dev_data, if_input_embedding, wordVector_file, if_subwords):
    dev_losses = []
    dev_accies = []
    iteration = 0
    middle_dev_evaluate = len(sen_arr) / 2
    # renew the computation graph
    dy.renew_cg()

    # define the parameters
    m = dy.ParameterCollection()

    w1 = m.add_parameters((N1, 250))
    w2 = m.add_parameters((len(tag_set), N1))
    b1 = m.add_parameters((N1))
    b2 = m.add_parameters((len(tag_set)))

    if if_input_embedding == 'embedding':
        with open(wordVector_file) as f:
            numbers = 0
            print "load word vectors ..."
            input_wordVectors = []
            for line in f:
                number_strings = line.split()  # Split the line on runs of whitespace
                numbers = [float(n) for n in number_strings]  # Convert to floats
                input_wordVectors.append(numbers)
            while len(input_wordVectors) < len(vocab):
                eps = np.sqrt(6)/np.sqrt(len(numbers))
                vec = np.random.uniform(-eps, eps, len(numbers))
                input_wordVectors.append(vec)
        E = m.add_lookup_parameters((len(vocab), len(input_wordVectors[0])))
        E.init_from_array(np.array(input_wordVectors))
    else:
        E = m.add_lookup_parameters((len(vocab), 50), init='uniform', scale=(np.sqrt(6)/np.sqrt(250)))

    # create trainer
    trainer = dy.AdamTrainer(m, alpha=LR)

    total_loss = 0
    seen_instances = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0
        correct =0
        train_counter = 0
        random.shuffle(sen_arr)
        for sen_i in range(len(sen_arr)):
            if sen_i % 1000 == 0:
                print("training sentance: %d" %sen_i)
            if sen_i > TRAIN_CUTOFF:
                continue
            if sen_i == middle_dev_evaluate:
                dev_loss, dev_acc = evaluate_dev(dev_data, (w1, w2, b1, b2, E, m), tag_set_rev, vocab)
                dev_losses.append(dev_loss)
                dev_accies.append(dev_acc)
            sen = sen_arr[sen_i]
            for i in range(len(sen)):
                dy.renew_cg()
                word, tag = sen[i].split()
                wordIndexVector = getVectorWordIndexes(i, sen, vocab)
                emb_vectors = [E[j] for j in wordIndexVector]
                con_emb_vectors = dy.concatenate(emb_vectors)
                if if_subwords == 'subwords':
                    net_input = add_subwords_vectors(E, con_emb_vectors, i, sen, vocab)
                else:
                    net_input = con_emb_vectors

                l1 = dy.tanh((w1*net_input)+b1)
                net_output = dy.softmax((w2*l1)+b2)
                y = int(tag_set[tag])

                loss = -(dy.log(dy.pick(net_output, y)))
                if REGULARIZATION_FACTOR > 0:
                    loss = loss + (dy.l2_norm(w1) * REGULARIZATION_FACTOR)

                seen_instances += 1
                train_counter += 1
                loss_val = loss.value()
                tag_hat = tag_set_rev[np.argmax(net_output.npvalue())]
                total_loss += loss_val
                epoch_loss += loss_val
                if tag_hat == tag:
                    if not (IGNORE_O and tag == 'O'):
                        correct += 1

                loss.backward()
                trainer.update()
                iteration += 1

        train_losses.append(epoch_loss/float(train_counter))
        train_acc.append(correct/float(train_counter))
        print ("Epoch: " + str(epoch+1) + "/" + str(EPOCHS) + \
            " average train loss is:", total_loss / seen_instances)
        dev_loss, dev_acc = evaluate_dev(dev_data, (w1, w2, b1, b2, E, m), tag_set_rev, vocab)
        dev_losses.append(dev_loss)
        dev_accies.append(dev_acc)

    return (w1, w2, b1, b2, E, m, dev_losses, dev_accies)


def add_subwords_vectors(E, con_emb_vectors, i, sen, vocab):
    preWordIndexVector, suffWordIndexVector = getVectorPreSuffWordIndexes(i, sen, vocab)
    pre_vectors = [E[j] for j in preWordIndexVector]
    suff_vectors = [E[j] for j in suffWordIndexVector]
    con_pre_vectors = dy.concatenate(pre_vectors)
    con_suff_vectors = dy.concatenate(suff_vectors)
    net_input = dy.esum([con_emb_vectors, con_pre_vectors, con_suff_vectors])
    return net_input


def evaluate_dev(dev_data, params, tag_set_rev, vocab):
    (w1, w2, b1, b2, E, m ) = params
    dev_sen_arr = load_sentences(dev_data)
    correct = 0
    wrong = 0
    counter = 0
    total_loss = 0

    for sen in dev_sen_arr:
        for i in range(len(sen)):
            word = sen[i].split()[0]
            tag = sen[i].split()[1]

            dy.renew_cg()

            vecs = getVectorWordIndexes(i, sen, vocab)
            emb_vectors = [E[j] for j in vecs]
            con_emb_vectors = dy.concatenate(emb_vectors)
            if if_subwords == 'subwords':
                net_input = add_subwords_vectors(E, con_emb_vectors, i, sen, vocab)
            else:
                net_input = con_emb_vectors

            l1 = dy.tanh((w1 * net_input) + b1)
            net_output = dy.softmax((w2 * l1) + b2)
            y = int(tag_set[tag])

            tag_hat = tag_set_rev[np.argmax(net_output.npvalue())]
            loss = -(dy.log(dy.pick(net_output, y)))
            total_loss += loss.value()

            counter +=1
            if tag_hat == tag:
                if not (IGNORE_O and tag == 'O'):
                    correct += 1
            else:
                wrong += 1

    total_loss /= float(counter)
    acc = correct / float(correct + wrong)
    print "\tDev accuracy is: %.2f" % (acc*100) + "%"

    return total_loss, acc


def predict_test(test_data, params, tag_set_rev, vocab):
    (w1, w2, b1, b2, E, m ) = params
    test_sen_arr = load_sentences(test_data)
    prediction = []

    for sen_i in range(len(test_sen_arr)):
        sen = test_sen_arr[sen_i]
        if sen_i % 100 == 0:
            print "Predict test sen " + str(sen_i + 1) + " of " + str(len(test_sen_arr))
        for i in range(len(sen)):
            word = sen[i].strip()
            dy.renew_cg()

            vecs = getVectorWordIndexes(i, sen, vocab)
            emb_vectors = [E[j] for j in vecs]
            con_emb_vectors = dy.concatenate(emb_vectors)
            if if_subwords == 'subwords':
                net_input = add_subwords_vectors(E, con_emb_vectors, i, sen, vocab)
            else:
                net_input = con_emb_vectors
            l1 = dy.tanh((w1 * net_input) + b1)
            net_output = dy.softmax((w2 * l1) + b2)

            tag_hat = tag_set_rev[np.argmax(net_output.npvalue())]

            prediction.append([word, tag_hat])
        prediction.append([])

    return prediction


def plotGraphs(dev_losses, dev_accies, if_input_embedding, data_type, if_subwords):
    if if_input_embedding == 'no-embedding':
        part = 'part1_'
    else:
        part = 'part3_'
    if if_subwords == 'subwords':
        part = str(part) + 'subwords_'

    # loss graph
    file_name = str(part) + str(data_type) + "_loss.png"
    plt.plot(dev_losses)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title(str(data_type))
    plt.legend()
    plt.savefig(file_name)
    plt.close()
    #plt.show()

    # acc graph
    file_name = str(part) + str(data_type) + "_acc.png"
    plt.plot(dev_accies)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title(str(data_type))
    plt.legend()
    plt.savefig(file_name)
    plt.close()
    #plt.show()


def write2file(prediction, if_input_embedding, data_type, if_subwords):
    if if_input_embedding == 'no-embedding':
        part = 'part1_'
    else:
        part = 'part3_'
    if if_subwords == 'subwords':
        part = str(part) + 'subwords_'
    file_name = str(part) + str(data_type) + "_test.pred"

    with open(file_name, 'w') as f:
        for i in range(len(prediction)):
            if len(prediction[i]) != 0:
                f.write("%s %s \n" % (prediction[i][0], prediction[i][1]))
            else:
                f.write("\n")


def usage():
    print "Input form should be:\n train_data dev_data test_data data_type [subwords/no-subwords] " \
          "[embedding/no-embedding] (if embedding) vocab_file wordVector_file"
    raise AssertionError()

if __name__ == '__main__':

    train_data = sys.argv[1]
    dev_data = sys.argv[2]
    test_data = sys.argv[3]
    data_type = sys.argv[4]
    if_subwords = sys.argv[5]
    if if_subwords != 'subwords' and if_subwords != 'no-subwords':
        usage()
    if_input_embedding = sys.argv[6]
    vocab_file = None
    wordVector_file = None
    if if_input_embedding == 'embedding':
        vocab_file = sys.argv[7]
        wordVector_file = sys.argv[8]
    elif if_input_embedding == 'no-embedding':
        pass
    else:
        usage()


    start_time = (datetime.datetime.now().time())

    vocab = getDataVocab(train_data, if_input_embedding, vocab_file, if_subwords)
    sen_arr = load_sentences(train_data)
    tag_set = load_tag_set(train_data) # tag_set is of form: "TAG NUM"
    tag_set_rev = reverse_tag_set(tag_set) # tag_set_rev is of form: "NUM TAG"

    (w1, w2, b1, b2, E, m, dev_losses, dev_accies) = train_model(sen_arr, vocab, tag_set, dev_data, if_input_embedding, wordVector_file, if_subwords)

    plotGraphs(dev_losses, dev_accies, if_input_embedding, "Dev_" + data_type, if_subwords)
    plotGraphs(train_losses, train_acc, if_input_embedding, "Train_" + data_type, if_subwords)

    prediction = predict_test(test_data, (w1, w2, b1, b2, E, m), tag_set_rev, vocab)
    write2file(prediction, if_input_embedding, data_type, if_subwords)

    print start_time
    print(datetime.datetime.now().time())

