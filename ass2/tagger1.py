import dynet_config
#dynet_config.set(random_seed=1)
import dynet as dy
import sys
import numpy as np
import random

#w1 = None
#w2 = None
#b1 = None
#b2 = None
#E = None

#word2vec = None
#tag_set = None

N1 = 1000
EPOCHS =5
LR = 0.0005

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


def reverse_tag_set(tag_set):
    tag_set_rev = {}
    for tag in tag_set:
        tag_set_rev[tag_set[tag]] = tag

    return tag_set_rev


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
    Converting train file to array of sentences
    :param train_data:
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
    trainer = dy.AdamTrainer(m, alpha=LR)

    total_loss = 0
    seen_instances = 0

    for epoch in range(EPOCHS):
        random.shuffle(sen_arr)
        for sen in sen_arr:
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
                #loss = loss + dy.l2_norm(m)
                print word, tag, tag_set_rev[np.argmax(net_output.npvalue())]
                seen_instances += 1
                total_loss += loss.value()

                loss.backward()
                trainer.update()

                if (seen_instances > 1 and seen_instances % 100 == 0):
                    print("average loss is:", total_loss / seen_instances)

            if sen_arr.index(sen) % 100 ==0:
                print "epoch: " +str(epoch) + " curr sentence: " + str(sen_arr.index(sen)) + "/" + str(len(sen_arr))

    return (w1, w2, b1, b2, E)


def evaluate_dev(dev_data, params, tag_set_rev, vocab):
    (w1, w2, b1, b2, E) = params
    dev_sen_arr = load_sentences(dev_data)
    correct = 0
    wrong = 0

    for sen in dev_sen_arr:
        #dy.renew_cg()
        print "Evaluate dev sen " + str(dev_sen_arr.index(sen) + 1) + " of " + str(len(dev_sen_arr))
        for i in range(len(sen)):
            dy.renew_cg()

            vecs = getVectorWordIndexes(i, sen, vocab)
            emb_vectors = [E[j] for j in vecs]

            net_input = dy.concatenate(emb_vectors)
            l1 = dy.tanh((w1 * net_input) + b1)
            net_output = dy.softmax((w2 * l1) + b2)

            tag_hat = tag_set_rev[np.argmax(net_output.npvalue())]

            word = sen[i].split()[0]
            tag = sen[i].split()[1]

            if tag_hat == tag:
                correct += 1
            else:
                wrong += 1
            #print word, tag, tag_hat

    acc = correct/float(correct+wrong) * 100
    print "\nDev accuracy is: %.2f" % (acc) + "%"



if __name__ == '__main__':

    train_data = sys.argv[1]
    dev_data = sys.argv[2]

    vocab = getDataVocab(train_data)
    sen_arr = load_sentences(train_data)
    tag_set = load_tag_set(train_data) # tag_set is of form: "TAG NUM"
    tag_set_rev = reverse_tag_set(tag_set) # tag_set_rev is of form: "NUM TAG"

    (w1, w2, b1, b2, E) = train_model(sen_arr, vocab, tag_set)

    evaluate_dev(dev_data, (w1, w2, b1, b2, E), tag_set_rev, vocab)



    pass
