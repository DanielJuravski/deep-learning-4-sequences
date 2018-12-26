import load_data
import dynet as dy
import numpy as np
from random import randint
from random import shuffle
import datetime
import sys
import re
import string


LEN_EMB_VECTOR = load_data.LEN_EMB_VECTOR
NUM_OF_OOV_EMBEDDINGS = load_data.NUM_OF_OOV_EMBEDDINGS
OOV_EMBEDDING_STR = load_data.OOV_EMBEDDING_STR

EPOCHS = 5
LR = 0.005

F_INPUT_SIZE = LEN_EMB_VECTOR
F_HIDDEN_SIZE = 200
F_OUTPUT_SIZE = 200

G_INPUT_SIZE = 2*LEN_EMB_VECTOR
G_HIDDEN_SIZE = 200
G_OUTPUT_SIZE = 200

H_INPUT_SIZE = 2*G_OUTPUT_SIZE
H_HIDDEN_SIZE = 200
H_OUTPUT_SIZE = 3



def init_model():
    model = dy.ParameterCollection()
    trainer = dy.AdagradTrainer(model, learning_rate=LR)
    model_params = {}

    # F feed-forward
    eps = np.sqrt(6) / np.sqrt(F_INPUT_SIZE + F_HIDDEN_SIZE)
    F_w1 = model.add_parameters((F_HIDDEN_SIZE, F_INPUT_SIZE),init='uniform', scale=eps)
    F_b1 = model.add_parameters((F_HIDDEN_SIZE))
    eps = np.sqrt(6) / np.sqrt(F_OUTPUT_SIZE + F_HIDDEN_SIZE)
    F_w2 = model.add_parameters((F_OUTPUT_SIZE, F_HIDDEN_SIZE),init='uniform', scale=eps)
    F_b2 = model.add_parameters((F_OUTPUT_SIZE))

    model_params['F_w1'] = F_w1
    model_params['F_b1'] = F_b1
    model_params['F_w2'] = F_w2
    model_params['F_b2'] = F_b2

    # G feed-forward
    eps = np.sqrt(6) / np.sqrt(G_HIDDEN_SIZE + G_INPUT_SIZE)
    G_w1 = model.add_parameters((G_HIDDEN_SIZE, G_INPUT_SIZE),init='uniform', scale=eps)
    G_b1 = model.add_parameters((G_HIDDEN_SIZE))
    eps = np.sqrt(6) / np.sqrt(G_OUTPUT_SIZE + G_HIDDEN_SIZE)
    G_w2 = model.add_parameters((G_OUTPUT_SIZE, G_HIDDEN_SIZE),init='uniform', scale=eps)
    G_b2 = model.add_parameters((G_OUTPUT_SIZE))

    model_params['G_w1'] = G_w1
    model_params['G_b1'] = G_b1
    model_params['G_w2'] = G_w2
    model_params['G_b2'] = G_b2

    # H feed-forward
    eps = np.sqrt(6) / np.sqrt(H_HIDDEN_SIZE + H_INPUT_SIZE)
    H_w1 = model.add_parameters((H_HIDDEN_SIZE, H_INPUT_SIZE),init='uniform', scale=eps)
    H_b1 = model.add_parameters((H_HIDDEN_SIZE))
    eps = np.sqrt(6) / np.sqrt(H_OUTPUT_SIZE + H_HIDDEN_SIZE)
    H_w2 = model.add_parameters((H_OUTPUT_SIZE, H_HIDDEN_SIZE),init='uniform', scale=eps)
    H_b2 = model.add_parameters((H_OUTPUT_SIZE))

    model_params['H_w1'] = H_w1
    model_params['H_b1'] = H_b1
    model_params['H_w2'] = H_w2
    model_params['H_b2'] = H_b2

    return model, model_params, trainer


def get_x_y(sample):
    sen1_str = sample[0]
    sen2_str = sample[1]
    label = sample[2]

    sen1 = sen1_str.split()
    sen2 = sen2_str.split()

    return sen1, sen2, label


def get_sen_from_dict(sen, emb_dict):
    emb_sen = []
    for word in sen:
        emb_word = get_word_from_dict(word, emb_dict)
        emb_sen.append(emb_word)

    return emb_sen


def get_word_from_dict(word, emb_dict):
    word = word.lower()
    word = word[:-1] + word[-1:].translate(None, string.punctuation)

    if emb_dict.has_key(word):
        word_emb = emb_dict[word]
    else:
        randoov = randint(0, NUM_OF_OOV_EMBEDDINGS-1)
        rand_word = OOV_EMBEDDING_STR + str(randoov)
        word_emb = emb_dict[rand_word]
        print "not in dict: " + word

    return word_emb


def set_E_matrix(sen1, sen2, len_sen1, len_sen2, model, model_params):
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
        F_x = dy.vecInput(F_INPUT_SIZE)
        F_x.set(word)
        F_i = (F_w2 * (dy.rectify(F_w1*F_x + F_b1)) + F_b2)
        F_sen1_list.append(F_i)
    F_sen1 = np.array(F_sen1_list)

    F_sen2_list = []
    for j in range(len_sen2):
        word = sen2[j]
        F_j = dy.vecInput(F_INPUT_SIZE)
        F_j.set(word)
        F_j = (F_w2 * (dy.rectify(F_w1*F_j + F_b1)) + F_b2)
        F_sen2_list.append(F_j)
    F_sen2 = np.array(F_sen2_list)

    for i in range(len_sen1):
        for j in range(len_sen2):
            e_ij = (dy.transpose(F_sen1[i])) * F_sen2[j]
            E_matrix[i][j] = e_ij.value()

    return E_matrix


def get_alpha_beta(E_matrix, len_cols, len_rows, sen1, sen2):
    """
    calculates alpha & beta from E, sen1 and sen2
    :param E_matrix:
    :param len_cols:
    :param len_rows:
    :param sen1:
    :param sen2:
    :param emb_data:
    :return: alpha and beta np array with size of sen2 and sen1 respectively * LEN_EMB_VECTOR (2d array)
    """
    sigma_exp_beta = np.array([])
    for i in range(len_cols):
        sigma_exp = 0
        for j in range(len_rows):
            sigma_exp += np.exp(E_matrix[i][j])
        sigma_exp_beta = np.append(sigma_exp_beta, sigma_exp)

    beta = []
    for j in range(len_rows):
        beta_i = 0
        for i in range(len_cols):
            beta_i += np.exp(E_matrix[i][j]) * (sen1[i]) / sigma_exp_beta[i]
        beta.append(beta_i)
    beta = np.array(beta)

    sigma_exp_alpha = np.array([])
    for j in range(len_rows):
        sigma_exp = 0
        for i in range(len_cols):
            sigma_exp += np.exp(E_matrix[i][j])
        sigma_exp_alpha = np.append(sigma_exp_alpha, sigma_exp)

    alpha = []
    for i in range(len_cols):
        alpha_i = 0
        for j in range(len_rows):
            alpha_i += np.exp(E_matrix[i][j]) * (sen2[j]) / sigma_exp_alpha[j]
        alpha.append(alpha_i)
    alpha = np.array(alpha)

    return alpha, beta


def get_v1_v2(beta, alpha, sen1, sen2, len_sen1, len_sen2, model, model_params):
    """

    :param beta:
    :param alpha:
    :param sen1:
    :param sen2:
    :param len_sen1:
    :param len_sen2:
    :param emb_data:
    :param model:
    :param model_params:
    :return: 2 lists of dynet expressions
    """
    G_w1 = model_params['G_w1']
    G_b1 = model_params['G_b1']
    G_w2 = model_params['G_w2']
    G_b2 = model_params['G_b2']

    v1 = []
    for i in range(len_sen1):
        beta_i = beta[i]
        con = sen1[i] + beta_i
        G_x = dy.vecInput(G_INPUT_SIZE)
        G_x.set(con)
        G_i = (G_w2 * (dy.rectify(G_w1 * G_x + G_b1)) + G_b2)
        v1.append(G_i)

    v2 = []
    for j in range(len_sen2):
        alpha_j = alpha[j]
        con = sen2[j] + alpha_j
        G_x = dy.vecInput(G_INPUT_SIZE)
        G_x.set(con)
        G_i = (G_w2 * (dy.rectify(G_w1 * G_x + G_b1)) + G_b2)
        v2.append(G_i)

    return v1, v2


def aggregate_v1_v2(v1, v2, model, model_params):
    """

    :param v1:
    :param v2:
    :param model:
    :param model_params:
    :return: y_hat softmaxed dynet expression
    """
    H_w1 = model_params['H_w1']
    H_b1 = model_params['H_b1']
    H_w2 = model_params['H_w2']
    H_b2 = model_params['H_b2']

    v1_esum = dy.esum(v1)
    v1_sum = v1_esum.value()

    v2_esum = dy.esum(v2)
    v2_sum = v2_esum.value()

    con = v1_sum + v2_sum

    H_x = dy.vecInput(H_INPUT_SIZE)
    H_x.set(con)

    y_hat = dy.softmax(H_w2 * (dy.rectify(H_w1 * H_x + H_b1)) + H_b2)

    return y_hat



def train_model(train_data, dev_data, emb_data, model, model_params, trainer):
    for epoch_i in range(EPOCHS):
        shuffle(train_data)
        print("Epoch " + str(epoch_i + 1) + " started at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        correct = wrong = 0.0
        total_loss = 0
        for sample_i in range(len(train_data)):
            #print sample_i
            dy.renew_cg()
            sample = train_data[sample_i]
            sen1_str, sen2_str, label = get_x_y(sample)
            len_sen1 = len(sen1_str)
            len_sen2 = len(sen2_str)
            sen1 = get_sen_from_dict(sen1_str, emb_data)
            sen2 = get_sen_from_dict(sen2_str, emb_data)
            #y = dy.scalarInput(0)
            #y.set(label)

            E_matrix = set_E_matrix(sen1, sen2, len_sen1, len_sen2, model, model_params)
            alpha, beta = get_alpha_beta(E_matrix, len_sen1, len_sen2, sen1, sen2)
            v1, v2 = get_v1_v2(alpha, beta, sen1, sen2, len_sen1, len_sen2, model, model_params)
            y_hat_vec_expression = aggregate_v1_v2(v1, v2, model, model_params)

            loss = -(dy.log(dy.pick(y_hat_vec_expression, label)))
            loss_val = loss.value()
            loss.backward()
            trainer.update()

            total_loss += loss_val
            label_hat = np.argmax(y_hat_vec_expression.npvalue())
            if label_hat == label:
                correct += 1
            else:
                wrong += 1

            if sample_i % 100 == 0 and (sample_i > 0):
                acc = (correct /(correct+wrong)) * 100
                relative_total_loss = total_loss/sample_i
                print("Epoch %d: Train iteration %d: total loss=%.4f loss=%.4f acc=%.2f%%" % (epoch_i+1, sample_i, relative_total_loss, loss_val, acc))





if __name__ == '__main__':
    if len(sys.argv) > 2:
        snli_train_file = sys.argv[1]
        glove_emb_file = sys.argv[2]
    else:
        snli_train_file = 'data/snli_1.0/snli_1.0_train.jsonl'
        snli_dev_file = 'data/snli_1.0/snli_1.0_dev.jsonl'
        snli_test_file = 'data/snli_1.0/snli_1.0_test.jsonl'

        glove_emb_file = 'data/glove/glove.6B.300d.txt'

    train_data = load_data.loadSNLI_labeled_data(snli_train_file)
    #dev_data = load_data.loadSNLI_labeled_data(snli_dev_file)
    dev_data = None
    #test_data = load_data.loadSNLI_labeled_data(snli_test_file)
    emb_data = load_data.get_emb_data(glove_emb_file)


    model, model_params, trainer = init_model()
    train_model(train_data, dev_data, emb_data, model, model_params, trainer)


    pass