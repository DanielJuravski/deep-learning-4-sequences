import load_data
import dynet as dy
import numpy as np
from random import randint
from random import shuffle
import datetime
import sys
import string
import matplotlib.pyplot as plt


LEN_EMB_VECTOR = load_data.LEN_EMB_VECTOR
NUM_OF_OOV_EMBEDDINGS = load_data.NUM_OF_OOV_EMBEDDINGS
OOV_EMBEDDING_STR = load_data.OOV_EMBEDDING_STR

EPOCHS = 5
LR = 0.001

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
        #print "not in dict: " + word

    word_dy_exp = dy.inputTensor(word_emb)
    return word_dy_exp


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
    E_matrix = []
    F_w1 = model_params['F_w1']
    F_b1 = model_params['F_b1']
    F_w2 = model_params['F_w2']
    F_b2 = model_params['F_b2']

    F_sen1 = []
    for i in range(len_sen1):
        F_i = dy.rectify(F_w2 * (dy.rectify((F_w1*sen1[i]) + F_b1)) + F_b2)
        F_sen1.append(F_i)

    F_sen2 = []
    for j in range(len_sen2):
        F_j = dy.rectify(F_w2 * (dy.rectify((F_w1*sen2[j]) + F_b1)) + F_b2)
        F_sen2.append(F_j)

    for i in range(len_sen1):
        E_row = []
        for j in range(len_sen2):
            e_ij = (dy.transpose(F_sen1[i])) * F_sen2[j]
            E_row.append(e_ij)
        E_matrix.append(E_row)

    return E_matrix


def get_alpha_beta(E_matrix, n_rows, n_cols, sen1, sen2):
    """
    calculates alpha & beta from E, sen1 and sen2
    :param E_matrix:
    :param n_rows:
    :param n_cols:
    :param sen1:
    :param sen2:
    :param emb_data:
    :return: alpha and beta np array with size of sen2 and sen1 respectively * LEN_EMB_VECTOR (2d array)
    """
    sigma_exp_beta = []
    for i in range(n_rows):
        sigma_exp = []
        for j in range(n_cols):
            sigma_exp.append(dy.exp(E_matrix[i][j]))
        sigma_exp_beta.append(dy.esum(sigma_exp))

    beta = []
    for i in range(n_rows):
        beta_i = []
        for j in range(n_cols):
            beta_i.append(dy.cmult((dy.cdiv(dy.exp(E_matrix[i][j]), sigma_exp_beta[i])), (sen2[j])))
        beta.append(dy.esum(beta_i))
    # beta = np.array(beta)
    #|beta| = nrows = |sen1|

    sigma_exp_alpha = []
    for j in range(n_cols):
        sigma_exp = []
        for i in range(n_rows):
            sigma_exp.append(dy.exp(E_matrix[i][j]))
        sigma_exp_alpha.append(dy.esum(sigma_exp))

    alpha = []
    for j in range(n_cols):
        alpha_j = []
        for i in range(n_rows):
            alpha_j.append(dy.cmult((dy.cdiv(dy.exp(E_matrix[i][j]), sigma_exp_alpha[j])), (sen1[i])))
        alpha.append(dy.esum(alpha_j))
    # alpha = np.array(alpha)
    #|alpha| = ncols = |sen2|

    return alpha, beta


def get_v1_v2(alpha, beta, sen1, sen2, len_sen1, len_sen2, model, model_params):
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
        con = dy.concatenate([sen1[i], beta_i])
        G_x = con
        G_i = dy.rectify(G_w2 * (dy.rectify((G_w1 * G_x) + G_b1)) + G_b2)
        v1.append(G_i)

    v2 = []
    for j in range(len_sen2):
        alpha_j = alpha[j]
        con = dy.concatenate([sen2[j], alpha_j])
        G_x = con
        G_i = dy.rectify(G_w2 * (dy.rectify((G_w1 * G_x) + G_b1)) + G_b2)
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
    v2_esum = dy.esum(v2)

    con = dy.concatenate([v1_esum, v2_esum])
    H_x = con

    y_hat = dy.softmax(H_w2 * (dy.rectify((H_w1 * H_x) + H_b1)) + H_b2)

    return y_hat



def train_model(train_data, emb_data, model, model_params, trainer):
    train_100_loss_val_list =[]
    train_100_acc_list = []
    correct = wrong = 0.0
    total_loss = 0
    for sample_i in range(len(train_data)):
        dy.renew_cg()
        sample = train_data[sample_i]
        sen1_str, sen2_str, label = get_x_y(sample)
        len_sen1 = len(sen1_str)
        len_sen2 = len(sen2_str)
        sen1 = get_sen_from_dict(sen1_str, emb_data)
        sen2 = get_sen_from_dict(sen2_str, emb_data)

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
            train_100_loss_val_list.append(relative_total_loss)
            train_100_acc_list.append(acc/100)
            print("Epoch %d: Train iteration %d: total-loss=%.4f loss=%.4f acc=%.2f%%" % (epoch_i+1, sample_i, relative_total_loss, loss_val, acc))

    epoch_loss =  sum(train_100_loss_val_list) / len(train_100_loss_val_list)
    epoch_acc = sum(train_100_acc_list) / len(train_100_acc_list)
    return epoch_loss, epoch_acc, train_100_loss_val_list, train_100_acc_list

def predict(data, emb_data, model, model_params):
    correct = wrong = 0.0
    total_loss = 0
    for sample_i in range(len(data)):
        dy.renew_cg()
        sample = train_data[sample_i]
        sen1_str, sen2_str, label = get_x_y(sample)
        len_sen1 = len(sen1_str)
        len_sen2 = len(sen2_str)
        sen1 = get_sen_from_dict(sen1_str, emb_data)
        sen2 = get_sen_from_dict(sen2_str, emb_data)

        E_matrix = set_E_matrix(sen1, sen2, len_sen1, len_sen2, model, model_params)
        alpha, beta = get_alpha_beta(E_matrix, len_sen1, len_sen2, sen1, sen2)
        v1, v2 = get_v1_v2(alpha, beta, sen1, sen2, len_sen1, len_sen2, model, model_params)
        y_hat_vec_expression = aggregate_v1_v2(v1, v2, model, model_params)

        loss = -(dy.log(dy.pick(y_hat_vec_expression, label)))
        loss_val = loss.value()

        total_loss += loss_val
        label_hat = np.argmax(y_hat_vec_expression.npvalue())
        if label_hat == label:
            correct += 1
        else:
            wrong += 1

    epoch_loss = total_loss / len(data)
    epoch_acc = correct / (correct+wrong)
    print("Dev Epoch %d: epoch-loss=%.4f acc=%.2f%%" % (epoch_i+1, epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def make_ststistics(loss_val_list, acc_list, prefix):
    plt_file_name = prefix + '_acc.png'
    plt.plot(acc_list)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.savefig(plt_file_name)
    plt.close()

    plt_file_name = prefix + '_loss.png'
    plt.plot(loss_val_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(plt_file_name)
    plt.close()


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
    dev_data = load_data.loadSNLI_labeled_data(snli_dev_file)
    #test_data = load_data.loadSNLI_labeled_data(snli_test_file)
    emb_data = load_data.get_emb_data(glove_emb_file)


    model, model_params, trainer = init_model()
    train_itreations_loss_val_list = []
    train_iterations_acc_list = []
    train_loss_val_list = []
    train_acc_list = []
    dev_loss_val_list = []
    dev_acc_list = []

    for epoch_i in range(EPOCHS):
        shuffle(train_data)
        print("Train Epoch " + str(epoch_i + 1) + " started at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        train_loss_val, train_acc, i_loss, i_acc = train_model(train_data, emb_data, model, model_params, trainer)
        train_loss_val_list.append(train_loss_val)
        train_acc_list.append(train_acc)
        train_itreations_loss_val_list += i_loss
        train_iterations_acc_list += i_acc
        shuffle(dev_data)
        print("Dev Epoch " + str(epoch_i + 1) + " started at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        dev_loss_val, dev_acc = predict(dev_data, emb_data, model, model_params)
        dev_loss_val_list.append(dev_loss_val)
        dev_acc_list.append(dev_acc)

    make_ststistics(train_itreations_loss_val_list, train_iterations_acc_list, "train_by_iteration")
    make_ststistics(train_loss_val_list, train_acc_list, "train_by_epoch")
    make_ststistics(dev_loss_val_list, dev_acc_list, "dev_by_epoch")
    print("finished at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



pass