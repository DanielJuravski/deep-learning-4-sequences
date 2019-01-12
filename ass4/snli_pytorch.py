import load_data
from random import randint
from random import shuffle
import datetime
import sys
import string
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


use_cuda = False

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

class AttentionModel(nn.Module):

    def __init__(self):
        super(AttentionModel, self).__init__()
        self.F_w1 = nn.Linear(F_INPUT_SIZE, F_HIDDEN_SIZE)
        self.F_w2 = nn.Linear(F_HIDDEN_SIZE, F_OUTPUT_SIZE)
        self.G_w1 = nn.Linear(G_INPUT_SIZE, G_HIDDEN_SIZE)
        self.G_w2 = nn.Linear(G_HIDDEN_SIZE, G_OUTPUT_SIZE)
        self.H_w1 = nn.Linear(H_INPUT_SIZE, H_HIDDEN_SIZE)
        self.H_w2 = nn.Linear(H_HIDDEN_SIZE, H_OUTPUT_SIZE)

    def forward(self, sen1, sen2):
        if use_cuda and torch.cuda.is_available():
            x_sen1 = [s1.cuda() for s1 in sen1]
            x_sen2 = [s2.cuda() for s2 in sen2]
        else:
            x_sen1 = sen1
            x_sen2 = sen2
        len_sen1 = len(x_sen1)
        len_sen2 = len(x_sen2)
        e_matrix = self.set_E_matrix(x_sen1, x_sen2, len_sen1, len_sen2)
        alpha, beta = self.get_alpha_beta(e_matrix, len_sen1, len_sen2, x_sen1, x_sen2)
        v1, v2 = self.get_v1_v2(alpha, beta, x_sen1, x_sen2, len_sen1, len_sen2)
        y_hat_vec_expression = self.aggregate_v1_v2(v1, v2)
        return y_hat_vec_expression

    def set_E_matrix(self, sen1, sen2, len_sen1, len_sen2):
        E_matrix = []

        F_sen1 = []
        for i in range(len_sen1):
            F_i = F.relu(self.F_w2(F.relu(self.F_w1(sen1[i].float()))))
            F_sen1.append(F_i)

        F_sen2 = []
        for j in range(len_sen2):
            F_j = F.relu(self.F_w2(F.relu(self.F_w1(sen2[j].float()))))
            F_sen2.append(F_j)

        for i in range(len_sen1):
            E_row = []
            for j in range(len_sen2):
                e_ij = F_sen1[i].dot(F_sen2[j])
                E_row.append(e_ij)
            E_matrix.append(E_row)

        return E_matrix


    def get_alpha_beta(self, E_matrix, n_rows, n_cols, sen1, sen2):
        sigma_exp_beta = []
        for i in range(n_rows):
            sigma_exp = torch.zeros(1)
            for j in range(n_cols):
                sigma_exp = sigma_exp + torch.exp(E_matrix[i][j])
            sigma_exp_beta.append(sigma_exp)

        beta = []
        for i in range(n_rows):
            beta_i = torch.zeros(1)
            for j in range(n_cols):
                beta_i = beta_i + (torch.mul((torch.div(torch.exp(E_matrix[i][j]), sigma_exp_beta[i])), (sen2[j].float())))
            beta.append(beta_i)
        # beta = np.array(beta)
        #|beta| = nrows = |sen1|

        sigma_exp_alpha = []
        for j in range(n_cols):
            sigma_exp = torch.zeros(1)
            for i in range(n_rows):
                sigma_exp = sigma_exp + (torch.exp(E_matrix[i][j]))
            sigma_exp_alpha.append(sigma_exp)

        alpha = []
        for j in range(n_cols):
            alpha_j = torch.zeros(1)
            for i in range(n_rows):
                alpha_j = alpha_j + (torch.mul((torch.div(torch.exp(E_matrix[i][j]), sigma_exp_alpha[j])), (sen1[i].float())))
            alpha.append(alpha_j)

        return alpha, beta

    def get_v1_v2(self, alpha, beta, sen1, sen2, len_sen1, len_sen2):
        v1 = []
        for i in range(len_sen1):
            beta_i = beta[i]
            G_x = torch.cat([sen1[i].float(), beta_i])
            G_i = F.relu(self.G_w2(F.relu(self.G_w1(G_x))))
            v1.append(G_i)

        v2 = []
        for j in range(len_sen2):
            alpha_j = alpha[j]
            G_x = torch.cat([sen2[j].float(), alpha_j])
            G_i = F.relu(self.G_w2(F.relu(self.G_w1(G_x))))
            v2.append(G_i)
        return v1, v2


    def aggregate_v1_v2(self, v1, v2):
        v1_esum = torch.zeros(1)
        for i in v1:
            v1_esum = v1_esum + i

        v2_esum = torch.zeros(1)
        for j in v2:
            v2_esum = v2_esum + j

        H_x = torch.cat([v1_esum, v2_esum])
        y_hat = F.softmax(self.H_w2(F.relu(self.H_w1(H_x))))

        return y_hat


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

    word_dy_exp = torch.from_numpy(word_emb)
    return word_dy_exp


def one_hot_from_label(label):
    v = torch.zeros(3).float()
    v[label] = 1.0
    return v.long()


def train_model(train_data, emb_data, model, optimizer):
    train_100_loss_val_list =[]
    train_100_acc_list = []
    correct = wrong = 0.0
    total_loss = 0

    model.train()
    for sample_i in range(len(train_data)):
        optimizer.zero_grad()
        sample = train_data[sample_i]
        sen1_str, sen2_str, label = get_x_y(sample)
        sen1 = get_sen_from_dict(sen1_str, emb_data)
        sen2 = get_sen_from_dict(sen2_str, emb_data)
        y_hat_vec_expression = model(sen1, sen2)
        one_hot_label = one_hot_from_label(label)
        loss = -torch.log(y_hat_vec_expression[label])
        loss_val = float(loss)
        pred = int(y_hat_vec_expression.data.argmax())
        loss.backward()
        optimizer.step()

        total_loss += loss_val
        if pred == label:
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

def predict(data, emb_data, model):
    correct = wrong = 0.0
    total_loss = 0
    model.eval()
    for sample_i in range(len(train_data)):
        # batch_sen1 = []
        # batch_sen2 = []
        # batch_y = []
        # if sample_i % 8 != 0:
        sample = train_data[sample_i]
        sen1_str, sen2_str, label = get_x_y(sample)
        sen1 = get_sen_from_dict(sen1_str, emb_data)
        sen2 = get_sen_from_dict(sen2_str, emb_data)
        #     batch_sen1.append(sen1)
        #     batch_sen2.append(sen2)
        #     batch_y.append(label)
        # else:

        y_hat_vec_expression = model(sen1, sen2)
        loss = -torch.log(y_hat_vec_expression[label])
        loss_val = float(loss)
        pred = int(y_hat_vec_expression.data.argmax())
        total_loss += loss_val
        if pred == label:
            correct += 1
        else:
            wrong += 1

    epoch_loss = total_loss / len(data)
    epoch_acc = correct / (correct+wrong)
    print("Epoch %d: epoch-loss=%.4f acc=%.2f%%" % (epoch_i+1, epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def make_ststistics(loss_val_list, acc_list, prefix):
    plt_file_name = basedir + prefix + '_acc.png'
    plt.plot(acc_list)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.savefig(plt_file_name)
    plt.close()

    plt_file_name = basedir + prefix + '_loss.png'
    plt.plot(loss_val_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(plt_file_name)
    plt.close()


if __name__ == '__main__':



    if len(sys.argv) > 3:
        snli_train_file = sys.argv[1]
        snli_dev_file = sys.argv[2]
        glove_emb_file = sys.argv[3]
    else:
        snli_train_file = 'data/snli_1.0/snli_1.0_train.jsonl'
        snli_dev_file = 'data/snli_1.0/snli_1.0_dev.jsonl'
        snli_test_file = 'data/snli_1.0/snli_1.0_test.jsonl'
        glove_emb_file = 'data/glove/glove.6B.300d.txt'

    if "--basedir" in sys.argv:
        basedir_arg = sys.argv.index("--basedir")
        basedir = sys.argv[basedir_arg +1]
    else:
        basedir = ""
    train_data = load_data.loadSNLI_labeled_data(snli_train_file)
    dev_data = load_data.loadSNLI_labeled_data(snli_dev_file)
    #test_data = load_data.loadSNLI_labeled_data(snli_test_file)
    emb_data = load_data.get_emb_data(glove_emb_file)

    model = AttentionModel()
    if use_cuda and torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=LR)
    train_itreations_loss_val_list = []
    train_iterations_acc_list = []
    train_loss_val_list = []
    train_acc_list = []
    dev_loss_val_list = []
    dev_acc_list = []

    for epoch_i in range(EPOCHS):
        shuffle(train_data)
        print("Train Epoch " + str(epoch_i + 1) + " started at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        train_loss_val, train_acc, i_loss, i_acc = train_model(train_data, emb_data, model, optimizer)
        train_loss_val_list.append(train_loss_val)
        train_acc_list.append(train_acc)
        train_itreations_loss_val_list += i_loss
        train_iterations_acc_list += i_acc
        shuffle(dev_data)
        print("Dev Epoch " + str(epoch_i + 1) + " started at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        dev_loss_val, dev_acc = predict(dev_data, emb_data, model)
        dev_loss_val_list.append(dev_loss_val)
        dev_acc_list.append(dev_acc)

    make_ststistics(train_itreations_loss_val_list, train_iterations_acc_list, "train_by_iteration")
    make_ststistics(train_loss_val_list, train_acc_list, "train_by_epoch")
    make_ststistics(dev_loss_val_list, dev_acc_list, "dev_by_epoch")
    print("finished at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



pass