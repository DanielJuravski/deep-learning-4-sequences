import dynet as dy
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime

UUUNKKK = 'UUUNKKK'

UNK_NUM = 'UNK_num'

UNK_ALLCAP = 'UNK_ALLCAP'

UNK_CAP_START = 'UNK_CapStart'

N1 = 50
EPOCHS = 5
LR = 0.0001
REGULARIZATION_FACTOR = 1
BATCH_SIZE = 1000


IGNORE_O = True
MIN_WORD_APPEARANCE = 1
TRAIN_CUTOFF = float("inf")

uncommon_words = set()


def getDataVocab(train_data, if_input_embedding, vocab_file):
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
    vocab[UUUNKKK] = size

    global uncommon_words
    if if_input_embedding == 'embedding':
        uncommon_words = set()
    else:
        for num, w in enumerate(word_count):
            if num <= MIN_WORD_APPEARANCE:
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


def train_model(sen_arr, vocab, tag_set, dev_data, if_input_embedding, wordVector_file):
    dev_losses = []
    dev_accies = []
    iteration = 0

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
    trainer = dy.AdamTrainer(m)

    total_loss = 0
    seen_instances = 0

    for epoch in range(EPOCHS):
        random.shuffle(sen_arr)
        for sen_i in range(len(sen_arr)):
            if sen_i % 1000 == 0:
                print("training sentance: %d" %sen_i)
            if sen_i > TRAIN_CUTOFF:
                continue
            sen = sen_arr[sen_i]
            for i in range(len(sen)):
                word, tag = sen[i].split()
                wordIndexVector = getVectorWordIndexes(i, sen, vocab)
                dy.renew_cg()
                emb_vectors = [E[j] for j in wordIndexVector]
                net_input = dy.concatenate(emb_vectors)
                l1 = dy.tanh((w1*net_input)+b1)
                net_output = dy.softmax((w2*l1)+b2)
                y = int(tag_set[tag])

                loss = -(dy.log(dy.pick(net_output, y)))
                #loss = loss + (dy.l2_norm(w1) * REGULARIZATION_FACTOR)

                seen_instances += 1
                loss_val = loss.value()
                total_loss += loss_val

                loss.backward()
                trainer.update()
                iteration += 1


        print ("Epoch: " + str(epoch+1) + "/" + str(EPOCHS) + \
            "average train loss is:", total_loss / seen_instances)
        dev_loss, dev_acc = evaluate_dev(dev_data, (w1, w2, b1, b2, E, m), tag_set_rev, vocab)
        dev_losses.append(dev_loss)
        dev_accies.append(dev_acc)

    return (w1, w2, b1, b2, E, m, dev_losses, dev_accies)


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

            net_input = dy.concatenate(emb_vectors)
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
    print "Dev accuracy is: %.2f" % (acc*100) + "%"

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

            net_input = dy.concatenate(emb_vectors)
            l1 = dy.tanh((w1 * net_input) + b1)
            net_output = dy.softmax((w2 * l1) + b2)

            tag_hat = tag_set_rev[np.argmax(net_output.npvalue())]

            prediction.append([word, tag_hat])
        prediction.append([])

    return prediction


def plotGraphs(dev_losses, dev_accies, if_input_embedding, data_type):
    if if_input_embedding == 'no-embedding':
        part = 'part1_'
    else:
        part = 'part3_'

    # loss graph
    file_name = str(part) + str(data_type) + "_loss.png"
    plt.plot(dev_losses)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('Dev Evaluation')
    plt.savefig(file_name)
    plt.close()
    #plt.show()

    # acc graph
    file_name = str(part) + str(data_type) + "_acc.png"
    plt.plot(dev_accies)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('Dev Evaluation')
    plt.savefig(file_name)
    #plt.show()


def write2file(prediction, if_input_embedding, data_type):
    if if_input_embedding == 'no-embedding':
        part = 'part1_'
    else:
        part = 'part3_'
    file_name = str(part) + str(data_type) + "_test.pred"

    with open(file_name, 'w') as f:
        for i in range(len(prediction)):
            if len(prediction[i]) != 0:
                f.write("%s %s \n" % (prediction[i][0], prediction[i][1]))
            else:
                f.write("\n")



if __name__ == '__main__':

    train_data = sys.argv[1]
    dev_data = sys.argv[2]
    test_data = sys.argv[3]
    data_type = sys.argv[4]
    if_input_embedding = sys.argv[5]
    vocab_file = None
    wordVector_file = None
    if if_input_embedding == 'embedding':
        vocab_file = sys.argv[6]
        wordVector_file = sys.argv[7]
    elif if_input_embedding == 'no-embedding':
        pass
    else:
        print "Input form should be:\n train_data dev_data test_data data_type [embedding/no-embedding] (if embedding) vocab_file wordVector_file"
        raise AssertionError()


    start_time = (datetime.datetime.now().time())


    vocab = getDataVocab(train_data, if_input_embedding, vocab_file)
    sen_arr = load_sentences(train_data)
    tag_set = load_tag_set(train_data) # tag_set is of form: "TAG NUM"
    tag_set_rev = reverse_tag_set(tag_set) # tag_set_rev is of form: "NUM TAG"

    (w1, w2, b1, b2, E, m, dev_losses, dev_accies) = train_model(sen_arr, vocab, tag_set, dev_data, if_input_embedding, wordVector_file)
    plotGraphs(dev_losses, dev_accies, if_input_embedding, data_type)

    prediction = predict_test(test_data, (w1, w2, b1, b2, E, m), tag_set_rev, vocab)
    write2file(prediction, if_input_embedding, data_type)

    print start_time
    print(datetime.datetime.now().time())

