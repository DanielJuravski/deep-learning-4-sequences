import load_data
import dynet as dy
import numpy as np
from random import randint

EPOCHS = 1

LEN_EMB_VECTOR = load_data.LEN_EMB_VECTOR
NUM_OF_OOV_EMBEDDINGS = load_data.NUM_OF_OOV_EMBEDDINGS
OOV_EMBEDDING_STR = load_data.OOV_EMBEDDING_STR

F_INPUT_SIZE = LEN_EMB_VECTOR
F_HIDDEN_SIZE = 200
F_OUTPUT_SIZE = 200



def init_model():
    model = dy.ParameterCollection()
    model_params = {}

    # F feed-forward
    F_w1 = model.add_parameters((F_HIDDEN_SIZE, F_INPUT_SIZE))
    F_b1 = model.add_parameters((F_HIDDEN_SIZE))
    F_w2 = model.add_parameters((F_OUTPUT_SIZE, F_HIDDEN_SIZE))
    F_b2 = model.add_parameters((F_OUTPUT_SIZE))

    model_params['F_w1'] = F_w1
    model_params['F_b1'] = F_b1
    model_params['F_w2'] = F_w2
    model_params['F_b2'] = F_b2



    return model, model_params


def get_x_y(sample):
    sen1_str = sample[0]
    sen2_str = sample[1]
    label = sample[2]

    sen1 = sen1_str.split()
    sen2 = sen2_str.split()

    return sen1, sen2, label


def get_word_from_dict(word, emb_dict):
    if emb_dict.has_key(word):
        word_emb = emb_dict[word]
    else:
        randoov = randint(0, NUM_OF_OOV_EMBEDDINGS-1)
        rand_word = OOV_EMBEDDING_STR + str(randoov)
        word_emb = emb_dict[rand_word]

    return word_emb


def set_E_matrix(sen1, sen2, len_sen1, len_sen2, model, model_params, emb_data):
    """

    :param sen1: str
    :param sen2: str
    :param len_sen1: int
    :param len_sen2: int
    :param model:
    :param model_params:
    :param emb_data:
    :return: matrix of np.zeroes
    """
    E_matrix = np.zeros((len_sen1, len_sen2))
    F_w1 = model_params['F_w1']
    F_b1 = model_params['F_b1']
    F_w2 = model_params['F_w2']
    F_b2 = model_params['F_b2']

    F_sen1_list = []
    for i in range(len_sen1):
        word = sen1[i]
        emb = get_word_from_dict(word, emb_data)
        F_x = dy.vecInput(F_INPUT_SIZE)
        F_x.set(emb)
        F_i = (F_w2 * (dy.rectify(F_w1*F_x + F_b1)) + F_b2)
        F_sen1_list.append(F_i)
    F_sen1 = np.array(F_sen1_list)

    F_sen2_list = []
    for j in range(len_sen2):
        word = sen2[j]
        emb = get_word_from_dict(word, emb_data)
        F_j = dy.vecInput(F_INPUT_SIZE)
        F_j.set(emb)
        F_j = (F_w2 * (dy.rectify(F_w1*F_j + F_b1)) + F_b2)
        F_sen2_list.append(F_j)
    F_sen2 = np.array(F_sen2_list)

    for i in range(len_sen1):
        for j in range(len_sen2):
            e_ij = (dy.transpose(F_sen1[i])) * F_sen2[j]
            E_matrix[i][j] = e_ij.value()

    return E_matrix


def train_model(train_data, dev_data, emb_data, model, model_params):
    for epoch_i in range(EPOCHS):
        for sample_i in range(len(train_data)):
            dy.renew_cg()
            sample = train_data[sample_i]
            sen1, sen2, label = get_x_y(sample)
            len_sen1 = len(sen1)
            len_sen2 = len(sen2)
            E_matrix = set_E_matrix(sen1, sen2, len_sen1, len_sen2, model, model_params, emb_data)

            pass




if __name__ == '__main__':
    snli_train_file = 'data/snli_1.0/snli_1.0_train.jsonl'
    snli_dev_file = 'data/snli_1.0/snli_1.0_dev.jsonl'
    snli_test_file = 'data/snli_1.0/snli_1.0_test.jsonl'

    train_data = load_data.loadSNLI_labeled_data(snli_train_file)
    #dev_data = load_data.loadSNLI_labeled_data(snli_dev_file)
    dev_data = None
    #test_data = load_data.loadSNLI_labeled_data(snli_test_file)

    glove_emb_file = 'data/glove/glove.6B.300d.txt'
    emb_data = load_data.get_emb_data(glove_emb_file)


    model, model_params = init_model()
    train_model(train_data, dev_data, emb_data, model, model_params)


    pass