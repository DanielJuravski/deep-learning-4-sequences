import dynet as dy
import sys
import numpy as np
import random

word2vec = None
tag_set = None

N1 = 500
EPOCHS = 10
BATCH_SIZE = 1000


def getDataVocab(train_data):
    vocab = {}
    size = 0
    with open(train_data) as f:
        for line in f:
            if line != '\n':
                word, tag = line.split()
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

    return vocab


def createWordVecDict(train_data):
    wordVec = {}
    eps = np.sqrt(6) / np.sqrt(50)
    with open(train_data) as f:
        for line in f:
            if line != '\n':
                word, tag = line.split()
                vector_value = np.random.uniform(low=-eps, high=eps, size=50)
                if not wordVec.has_key(word):
                    wordVec[word] = vector_value

    return wordVec


def load_wordVectorsVocab(vocab_data, wordVectors_data):
    vocab_vector = {}

    with open(vocab_data) as vocab_f, open(wordVectors_data) as wordVector_f:
        vocab_f_num_lines = sum(1 for line in vocab_f)
        wordVector_f_num_lines = sum(1 for line in wordVector_f)
        if vocab_f_num_lines != wordVector_f_num_lines:
            print "Number of lines in %s and %s is not identical !!!" % (vocab_data, wordVectors_data)
            raise AssertionError()
        # define file pointer to the head of the file
        vocab_f.seek(0)
        wordVector_f.seek(0)

        for word, tag in zip(vocab_f, wordVector_f):
            vocab_vector[word.strip()] = tag

    print "Word to vector dictionary was loaded."
    return vocab_vector


def load_train_data (train_data):
    """
    Converting train file to array of sentences, and getting the possible labels of the tagset.
    :param train_data:
    :return: sen_arr, tag_set
    """
    train_f = open(train_data)
    sen_arr = []
    sen = []
    tag_set = {}
    num = 0
    for line in train_f:
        if line != '\n':
            word, tag = line.split()
            if tag not in tag_set:
                tag_set[tag] = num
                num += 1
            sen.append(line)
        else:
            sen_arr.append(sen)
            sen = []
    sen_arr.append(sen)
    train_f.close()

    return sen_arr, tag_set


def getWordIndex(sen, k):
    """
    get word's index from the vocab
    :param sen, k
    :return:
    """
    i = vocab[sen[k].split()[0]]

    return i


def getVectorWordIndexes(i, sen, vocab):
    sen_len = len(sen)

    if i < 2:
        wpp = vocab['/S/S']
    else:
        wpp = getWordIndex(sen, i-2)

    if i < 1:
        wp = vocab['/S']
    else:
        wp = getWordIndex(sen, i-1)

    wi = getWordIndex(sen, i)

    if i > sen_len - 2:
        wn = vocab['/E']
    else:
        wn = getWordIndex(sen, i+1)

    if i > sen_len - 3:
        wnn = vocab['/E/E']
    else:
        wnn = getWordIndex(sen, i+2)

    total_i = [wpp, wp, wi, wn, wnn]

    return total_i


def train_model(sen_arr, vocab, tag_set):
    # renew the computation graph
    dy.renew_cg()

    # define the parameters
    m = dy.ParameterCollection()
    w1 = m.add_parameters((N1, 250))
    w2 = m.add_parameters((len(tag_set), N1))
    b1 = m.add_parameters((N1))
    b2 = m.add_parameters((len(tag_set)))
    E = m.add_lookup_parameters((len(vocab), 50), init='uniform', scale=(np.sqrt(6)/np.sqrt(50)))

    # create trainer
    trainer = dy.SimpleSGDTrainer(m)

    total_loss = 0
    seen_instances = 0

    for epoch in range(EPOCHS):
        random.shuffle(sen_arr)
        for sen_i in range(BATCH_SIZE):
            sen = sen_arr[sen_i]
            for i in range(len(sen)):
                dy.renew_cg()
                word, tag = sen[i].split()
                wordIndexVector = getVectorWordIndexes(i, sen, vocab)
                emb_vectors = [E[j] for j in wordIndexVector]
                net_input = dy.concatenate(emb_vectors)
                l1 = dy.tanh((w1*net_input)+b1)
                net_output = (w2*l1)+b2

                y = int(tag_set[tag])

                loss = dy.pickneglogsoftmax(net_output, y)

                seen_instances += 1
                total_loss += loss.value()

                loss.backward()
                trainer.update()

                if (seen_instances > 1 and seen_instances % 1000 == 0):
                    print("average loss is:", total_loss / seen_instances)
            if sen_arr.index(sen) % 100 ==0:
                print str(sen_arr.index(sen)) + "/" + str(len(sen_arr))




if __name__ == '__main__':

    train_data = sys.argv[1]
    #vocab_data = sys.argv[2]
    #wordVectors_data = sys.argv[3]

    #word2vec = createWordVecDict(train_data)
    vocab = getDataVocab(train_data)
    sen_arr, tag_set = load_train_data(train_data)

    train_model(sen_arr, vocab, tag_set)

    pass
