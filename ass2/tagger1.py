import sys
import numpy as np
#import dynet

word2vec = None


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
    Convertinr train file to array of sentences
    :param train_data:
    :return: sen_arr
    """
    train_f = open(train_data)
    sen_arr = []
    sen = []
    for line in train_f:
        if line != '\n':
            sen.append(line)
        else:
            sen_arr.append(sen)
            sen = []
    sen_arr.append(sen)
    train_f.close()

    return sen_arr

def getWordVector(pair):
    """

    :param pair: WORD TAG
    :return:
    """
    word, tag = pair.split()
    word = word.lower()
    vec = word2vec[word]
    vec = vec.split()
    vec = [float(x) for x in vec]
    vec = np.array(vec)

    return vec


def getVectorByWord(i, sen):
    sen_len = len(sen)

    if i < 2:
        vec_pp = np.zeros(50)
    else:
        vec_pp = getWordVector(sen[i-2])

    if i < 1:
        vec_p = np.zeros(50)
    else:
        vec_p = getWordVector(sen[i-1])

    vec_i = getWordVector(sen[i])

    if i > sen_len - 2:
        vec_n = np.zeros(50)
    else:
        vec_n = getWordVector(sen[i+1])

    if i > sen_len - 3:
        vec_nn = np.zeros(50)
    else:
        vec_nn = getWordVector(sen[i+2])

    total_vec = np.concatenate((vec_pp, vec_p, vec_i, vec_n, vec_nn))

    return total_vec


def train_model(sen_arr):
    for sen in sen_arr:
        for i in range(len(sen)):
            word, tag = sen[i].split()
            wordVector = getVectorByWord(i, sen)


    pass


if __name__ == '__main__':
    #option = sys.argv[1]
    #if option == '-train':

    train_data = sys.argv[1]
    vocab_data = sys.argv[2]
    wordVectors_data = sys.argv[3]

    word2vec = load_wordVectorsVocab(vocab_data, wordVectors_data)
    sen_arr = load_train_data(train_data)


    train_model(sen_arr)
