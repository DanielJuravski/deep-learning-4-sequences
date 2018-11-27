import dynet as dy
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime


N1 = 1000
EPOCHS = 3
LR = 0.0005

BATCH_SIZE = 1000


def getDataVocab(vocab_file):
    vocab = {}
    size = 0
    with open(vocab_file) as f:
        for word in f:
            word = word.rstrip()
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


def getWordIndex(sen, k, vocab):
    """
    get word's index from the vocab
    :param sen, k
    :return:
    """
    word = sen[k].split()[0]
    word = word.lower()
    if vocab.has_key(word):
        i = vocab[word]
    else:
        i = vocab['UUUNKKK']
        #print word

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


def train_model(sen_arr, vocab, tag_set, wordVector_file, dev_data):
    dev_losses = []
    dev_accies = []
    iteration = 0
    with open(wordVector_file) as f:
        print "load word vectors ..."
        input_wordVectors = []
        for line in f:
            number_strings = line.split()  # Split the line on runs of whitespace
            numbers = [float(n) for n in number_strings]  # Convert to floats
            input_wordVectors.append(numbers)

    # renew the computation graph
    dy.renew_cg()

    # define the parameters
    m = dy.ParameterCollection()

    w1 = m.add_parameters((N1, 250))
    w2 = m.add_parameters((len(tag_set), N1))
    b1 = m.add_parameters((N1))
    b2 = m.add_parameters((len(tag_set)))
    E = m.add_lookup_parameters((len(input_wordVectors)+4, len(input_wordVectors[0])))
    E.init_from_array(np.array(input_wordVectors))

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
                seen_instances += 1
                loss_val = loss.value()
                total_loss += loss_val


                loss.backward()
                trainer.update()
                iteration += 1

                if (iteration % 10000 == 0):
                    print "Epoch: " +str(epoch) + "/" + str(EPOCHS) + " Sentence: " + str(sen_arr.index(sen)) + "/" + str(len(sen_arr)), \
                          "average loss is:", total_loss / seen_instances, "iteration loss: ", loss_val
                    loss, acc = evaluate_dev(dev_data, (w1, w2, b1, b2, E, m), tag_set_rev, vocab)
                    dev_losses.append(loss)
                    dev_accies.append(acc)

    return w1, w2, b1, b2, E, m, dev_losses, dev_accies


def evaluate_dev(dev_data, params, tag_set_rev, vocab):
    (w1, w2, b1, b2, E, m ) = params
    dev_sen_arr = load_sentences(dev_data)
    correct = 0
    wrong = 0
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

            if tag_hat == tag:
                correct += 1
            else:
                wrong += 1

    total_loss /= float(correct + wrong)
    acc = correct / float(correct + wrong)
    print "Dev accuracy is: %.2f" % (acc * 100) + "%"

    return total_loss, acc


def plotGraphs(dev_losses, dev_accies):
    plt.plot(dev_losses)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('Dev Evaluation')
    plt.savefig('tagger2_loss.png')
    #plt.show()

    plt.plot(dev_accies)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('Dev Evaluation')
    plt.savefig('tagger2_acc.png')
    #plt.show()


def predict_test(test_data, params, tag_set_rev, vocab):
    (w1, w2, b1, b2, E, m ) = params
    test_sen_arr = load_sentences(test_data)
    prediction = []

    for sen in test_sen_arr:
        print "Predict test sen " + str(test_sen_arr.index(sen) + 1) + " of " + str(len(test_sen_arr))
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


def write2file(prediction):
    with open('tagger2_test_pred', 'w') as f:
        for i in range(len(prediction)):
            if len(prediction[i]) != 0:
                f.write("%s %s \n" % (prediction[i][0], prediction[i][1]))
            else:
                f.write("\n")


if __name__ == '__main__':

    train_data = sys.argv[1]
    dev_data = sys.argv[2]
    test_data = sys.argv[3]
    vocab_file = sys.argv[4]
    wordVector_file = sys.argv[5]

    print(datetime.datetime.now().time())

    vocab = getDataVocab(vocab_file)
    sen_arr = load_sentences(train_data)
    tag_set = load_tag_set(train_data) # tag_set is of form: "TAG NUM"
    tag_set_rev = reverse_tag_set(tag_set) # tag_set_rev is of form: "NUM TAG"

    (w1, w2, b1, b2, E, m, dev_losses, dev_accies) = train_model(sen_arr, vocab, tag_set, wordVector_file, dev_data)
    plotGraphs(dev_losses, dev_accies)

    prediction = predict_test(test_data, (w1, w2, b1, b2, E, m), tag_set_rev, vocab)
    write2file(prediction)

    print(datetime.datetime.now().time())


