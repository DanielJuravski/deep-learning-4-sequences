import load_data
import dynet as dy
import dynet_config
dynet_config.set(autobatch=1)
import numpy as np
from random import randint
import datetime
import sys
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle

LEN_EMB_VECTOR = load_data.LEN_EMB_VECTOR
NUM_OF_OOV_EMBEDDINGS = load_data.NUM_OF_OOV_EMBEDDINGS
OOV_EMBEDDING_STR = load_data.OOV_EMBEDDING_STR

EPOCHS = 15
LR = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 16

F_INPUT_SIZE = LEN_EMB_VECTOR
F_HIDDEN_SIZE = 200
F_OUTPUT_SIZE = 200

G_INPUT_SIZE = 2*F_OUTPUT_SIZE
G_HIDDEN_SIZE = 200
G_OUTPUT_SIZE = 200

H_INPUT_SIZE = 2*G_OUTPUT_SIZE
H_HIDDEN_SIZE = 200
H_OUTPUT_SIZE = 3



def init_model(emb_dict, emb):
    model = dy.ParameterCollection()
    #trainer = dy.AdagradTrainer(model, learning_rate=LR)
    trainer = dy.AdamTrainer(model, alpha=LR)

    model_params = {}

    # F feed-forward
    eps = np.sqrt(6) / np.sqrt(F_INPUT_SIZE + F_HIDDEN_SIZE)
    F_w1 = model.add_parameters((F_HIDDEN_SIZE, F_INPUT_SIZE),init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(F_HIDDEN_SIZE)
    F_b1 = model.add_parameters((F_HIDDEN_SIZE), init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(F_OUTPUT_SIZE + F_HIDDEN_SIZE)
    F_w2 = model.add_parameters((F_OUTPUT_SIZE, F_HIDDEN_SIZE),init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(F_OUTPUT_SIZE)
    F_b2 = model.add_parameters((F_OUTPUT_SIZE), init='uniform', scale=eps)

    model_params['F_w1'] = F_w1
    model_params['F_b1'] = F_b1
    model_params['F_w2'] = F_w2
    model_params['F_b2'] = F_b2

    # G feed-forward
    eps = np.sqrt(6) / np.sqrt(G_HIDDEN_SIZE + G_INPUT_SIZE)
    G_w1 = model.add_parameters((G_HIDDEN_SIZE, G_INPUT_SIZE), init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(G_HIDDEN_SIZE)
    G_b1 = model.add_parameters((G_HIDDEN_SIZE), init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(G_OUTPUT_SIZE + G_HIDDEN_SIZE)
    G_w2 = model.add_parameters((G_OUTPUT_SIZE, G_HIDDEN_SIZE), init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(G_OUTPUT_SIZE)
    G_b2 = model.add_parameters((G_OUTPUT_SIZE), init='uniform', scale=eps)

    model_params['G_w1'] = G_w1
    model_params['G_b1'] = G_b1
    model_params['G_w2'] = G_w2
    model_params['G_b2'] = G_b2

    # H feed-forward
    eps = np.sqrt(6) / np.sqrt(H_HIDDEN_SIZE + H_INPUT_SIZE)
    H_w1 = model.add_parameters((H_HIDDEN_SIZE, H_INPUT_SIZE), init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(H_HIDDEN_SIZE)
    H_b1 = model.add_parameters((H_HIDDEN_SIZE), init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(H_OUTPUT_SIZE + H_HIDDEN_SIZE)
    H_w2 = model.add_parameters((H_OUTPUT_SIZE, H_HIDDEN_SIZE), init='uniform', scale=eps)
    eps = np.sqrt(6) / np.sqrt(H_HIDDEN_SIZE)
    H_b2 = model.add_parameters((H_OUTPUT_SIZE), init='uniform', scale=eps)

    model_params['H_w1'] = H_w1
    model_params['H_b1'] = H_b1
    model_params['H_w2'] = H_w2
    model_params['H_b2'] = H_b2

    len_of_emb = len(emb_dict)
    print "Len of vocab: " + str(len_of_emb)
    E = model.add_lookup_parameters((len_of_emb, 300))
    E.init_from_array(np.array(emb))

    model_params['E'] = E

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
        if '-' in word:
            word = word.split('-')
        else:
            word = [word]
        for word_i in word:
            emb_word = get_word_from_dict(word_i, emb_dict)
            emb_sen.append(emb_word)

    return emb_sen


def word_dash_found(word, emb_dict):
    word_emb = False
    if '-' in word:
        words = word.split('-')
        for word in words:
            if emb_dict.has_key(word):
                word_emb = emb_dict[word]
                break

    return word_emb


def set_E_matrix(sen1, sen2, model_params):
    F_w1 = model_params['F_w1']
    F_b1 = model_params['F_b1']
    F_w2 = model_params['F_w2']
    F_b2 = model_params['F_b2']

    #sen1 = dy.dropout(sen1, DROPOUT_RATE)
    #sen2 = dy.dropout(sen2, DROPOUT_RATE)

    F_sen1 = dy.rectify(F_w2 * (dy.rectify(dy.colwise_add(F_w1*sen1, F_b1))) + F_b2)
    F_sen2 = dy.rectify(F_w2 * (dy.rectify(dy.colwise_add(F_w1*sen2, F_b1))) + F_b2)

    E_matrix = (dy.transpose(F_sen1)) * F_sen2

    return E_matrix, F_sen1, F_sen2


def get_alpha_beta(E_matrix, F_sen1, F_sen2):
    alpha_softmax = dy.softmax(E_matrix)
    beta_softmax = dy.softmax(dy.transpose(E_matrix))

    beta = F_sen2 * dy.transpose(alpha_softmax)
    alpha = F_sen1 * dy.transpose(beta_softmax)

    return alpha, beta


def get_v1_v2(alpha, beta, sen1, sen2, model_params):
    G_w1 = model_params['G_w1']
    G_b1 = model_params['G_b1']
    G_w2 = model_params['G_w2']
    G_b2 = model_params['G_b2']

    con = dy.concatenate([sen1, beta], d=0)
    #con = dy.dropout(con, DROPOUT_RATE)
    v1 = dy.rectify(G_w2 * (dy.rectify(dy.colwise_add(G_w1 * con, G_b1))) + G_b2)

    con = dy.concatenate([sen2, alpha], d=0)
    #con = dy.dropout(con, DROPOUT_RATE)
    v2 = dy.rectify(G_w2 * (dy.rectify(dy.colwise_add(G_w1 * con, G_b1))) + G_b2)

    return v1, v2


def aggregate_v1_v2(v1, v2, model_params):
    H_w1 = model_params['H_w1']
    H_b1 = model_params['H_b1']
    H_w2 = model_params['H_w2']
    H_b2 = model_params['H_b2']

    v1_sum = dy.sum_dim(v1, [1])
    v2_sum = dy.sum_dim(v2, [1])

    con = dy.concatenate([v1_sum, v2_sum])
    #con = dy.dropout(con, DROPOUT_RATE)

    y_hat = dy.softmax(H_w2 * (dy.rectify((H_w1 * con) + H_b1)) + H_b2)

    return y_hat


def get_emb_i(word):
    original_word = word
    if emb_dict.has_key(word):
        w_i = emb_dict[word]
    else:
        # lowercase word
        word = word.lower()
        if emb_dict.has_key(word):
            w_i = emb_dict[word]
        else:
            # remove any punctuation
            tuned_word = ''.join(letter for letter in word if letter.isalpha())
            if emb_dict.has_key(tuned_word):
                w_i = emb_dict[tuned_word]
            else:
                wordnet_lemmatizer = WordNetLemmatizer()
                word = wordnet_lemmatizer.lemmatize(tuned_word)
                if emb_dict.has_key(word):
                    w_i = emb_dict[word]
                    #print "LEMMA"
                else:
                    lancaster_stemmer = LancasterStemmer()
                    word = lancaster_stemmer.stem(tuned_word)
                    if emb_dict.has_key(word):
                        w_i = emb_dict[word]
                        #print "STEMMER"
                    else:
                        porter = PorterStemmer()
                        word = str(porter.stem(tuned_word))
                        if emb_dict.has_key(word):
                            w_i = emb_dict[word]
                            #print "PORTER"
                        else:
                            if word in oov_dict:
                                word_oov_i = oov_dict[word]
                                w_i = emb_dict[word_oov_i]
                            else:
                                randoov = randint(0, NUM_OF_OOV_EMBEDDINGS - 1)
                                word_oov_i = OOV_EMBEDDING_STR + str(randoov)
                                oov_dict[word] = word_oov_i
                                w_i = emb_dict[word_oov_i]
                            # if word == " " or word == "":
                            #     print "not in dict: " + word + "ORIGINAL WORD: " + original_word
                            # else:
                            #     print "not in dict: " + word

    return w_i


def embed(sentence, model, model_params, trainer):
    E = model_params['E']
    sentence_embedded = [dy.lookup(E, get_emb_i(w), update=False) for w in sentence.split()]
    sentence_embedded = dy.concatenate(sentence_embedded, d=1)
    return sentence_embedded


def feed_farword(sen1, sen2, model, model_params, trainer):
    sen1_emb = embed(sen1, model, model_params, trainer)
    sen2_emb = embed(sen2, model, model_params, trainer)

    E_matrix, F_sen1, F_sen2 = set_E_matrix(sen1_emb, sen2_emb, model_params)
    alpha, beta = get_alpha_beta(E_matrix, F_sen1, F_sen2)
    v1, v2 = get_v1_v2(alpha, beta, F_sen1, F_sen2, model_params)
    y_hat_vec_expression = aggregate_v1_v2(v1, v2, model_params)

    return y_hat_vec_expression


def train_model(train_data, model, model_params, trainer, dev_data):
    train_src_data, train_target_data, train_label_data = train_data
    dev_src_data, dev_target_data, dev_label_data = dev_data

    train_correct = train_wrong = 0.0
    train_correct_i = train_wrong_i = 0.0
    total_acc = 0
    train_total_loss = 0
    shift = 0
    i = 0
    train_acc_list = []
    train_loss_list = []
    dev_acc_list = []
    dev_loss_list = []
    len_of_train_data = len(train_src_data)

    while shift < len_of_train_data:
        losses = []
        dy.renew_cg()
        sen1_batch = train_src_data[shift: shift+BATCH_SIZE]
        sen2_batch = train_target_data[shift: shift+BATCH_SIZE]
        label_batch = train_label_data[shift: shift+BATCH_SIZE]

        for sen1, sen2, label in zip(sen1_batch, sen2_batch, label_batch):
            y_hat_vec_expression = feed_farword(sen1, sen2, model, model_params, trainer)
            loss = -(dy.log(dy.pick(y_hat_vec_expression, label)))
            losses.append(loss)
            label_hat = np.argmax(y_hat_vec_expression.npvalue())
            if label_hat == label:
                train_correct_i += 1
            else:
                train_wrong_i += 1

        loss_batch = dy.esum(losses) / BATCH_SIZE
        i += 1
        train_total_loss += loss_batch.scalar_value()
        loss_batch.backward()
        trainer.update()
        shift += BATCH_SIZE

        if i % (500 // BATCH_SIZE) == 0:
            acc_i = (train_correct_i / (train_correct_i+train_wrong_i)) * 100
            train_correct += train_correct_i
            train_wrong += train_wrong_i
            total_acc = (train_correct / (train_correct + train_wrong)) * 100
            train_correct_i = train_wrong_i = 0.0
            loss_val = train_total_loss/i
            train_acc_list.append(acc_i/100)
            train_loss_list.append(loss_val)
            print("Epoch %d: Train iteration %d/%d: loss=%.4f iteration_acc=%.2f%% total acc=%.2f%%" % (epoch_i+1, shift, len_of_train_data, loss_val, acc_i, total_acc))

        if i % (50000 // BATCH_SIZE) == 0:
            dev_loss = 0
            dev_correct = dev_wrong = 0.0
            len_dev_data = len(dev_src_data)

            for sen1, sen2, label in zip(dev_src_data, dev_target_data, dev_label_data):
                dy.renew_cg()
                y_hat_vec_expression = feed_farword(sen1, sen2, model, model_params, trainer)
                loss = -(dy.log(dy.pick(y_hat_vec_expression, label)))
                dev_loss += loss.value()
                label_hat = np.argmax(y_hat_vec_expression.npvalue())
                if label_hat == label:
                    dev_correct += 1
                else:
                    dev_wrong += 1

            dev_acc = (dev_correct / (dev_correct + dev_wrong)) * 100
            loss_val = dev_loss / len_dev_data
            dev_acc_list.append(dev_acc/100)
            dev_loss_list.append(loss_val)
            print("==== Epoch %d: DEV loss=%.4f acc=%.2f%% ====" % (epoch_i + 1, loss_val, dev_acc))

    stats = (train_acc_list, train_loss_list, dev_acc_list, dev_loss_list)
    return model, model_params, stats


def make_statistics(loss_val_list, acc_list, prefix):
    title = ''
    if prefix == 'train':
        title = 'Train'
    elif prefix == 'dev':
        title = 'Dev'

    plt_file_name = prefix + '_acc.png'
    plt.plot(acc_list)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title+' Accuracy')
    plt.savefig(plt_file_name)
    plt.close()

    plt_file_name = prefix + '_loss.png'
    plt.plot(loss_val_list)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title+' Loss')
    plt.savefig(plt_file_name)
    plt.close()


def predict_test(test_data, model, model_params, trainer):
    test_src_data, test_target_data, test_label_data = test_data
    test_loss = 0
    test_correct = test_wrong = 0.0
    len_test_data = len(test_src_data)
    for sen1, sen2, label in zip(test_src_data, test_target_data, test_label_data):
        dy.renew_cg()
        y_hat_vec_expression = feed_farword(sen1, sen2, model, model_params, trainer)
        loss = -(dy.log(dy.pick(y_hat_vec_expression, label)))
        test_loss += loss.value()
        label_hat = np.argmax(y_hat_vec_expression.npvalue())
        if label_hat == label:
            test_correct += 1
        else:
            test_wrong += 1

    test_acc = (test_correct / (test_correct + test_wrong)) * 100
    loss_val = test_loss / len_test_data

    return test_acc, loss_val


if __name__ == '__main__':
    if len(sys.argv) > 3:
        snli_train_file = sys.argv[1]
        snli_dev_file = sys.argv[2]
        snli_test_file = sys.argv[3]
        glove_emb_file = sys.argv[4]
        nltk_data = sys.argv[5]
    else:
        wrong_argument_err = "You need to insert 4 arguments:\n$ python snli.py <train_data> <dev_data> <test_data> <glove_data>"
        raise Exception(wrong_argument_err)
        #snli_train_file = 'data/snli_1.0/snli_1.0_train.jsonl'
        #snli_dev_file = 'data/snli_1.0/snli_1.0_dev.jsonl'
        #snli_test_file = 'data/snli_1.0/snli_1.0_test.jsonl'
        #glove_emb_file = 'data/glove/glove.6B.300d.txt'
        #nltk_data = "nltk_data/"

    train_src_data, train_target_data, train_label_data = load_data.loadSNLI_labeled_data(snli_train_file)
    dev_src_data, dev_target_data, dev_label_data = load_data.loadSNLI_labeled_data(snli_dev_file)
    test_src_data, test_target_data, test_label_data = load_data.loadSNLI_labeled_data(snli_test_file)
    emb_dict, emb = load_data.get_emb_data(glove_emb_file)
    oov_dict = {}
    nltk.data.path.append(nltk_data)

    model, model_params, trainer = init_model(emb_dict, emb)

    train_acc_list = []
    train_loss_list = []
    dev_acc_list = []
    dev_loss_list = []

    for epoch_i in range(EPOCHS):
        train_src_data, train_target_data, train_label_data = shuffle(train_src_data, train_target_data, train_label_data)
        train_data = train_src_data, train_target_data, train_label_data
        dev_data = dev_src_data, dev_target_data, dev_label_data
        print("Epoch " + str(epoch_i + 1) + " started at: " + datetime.datetime.now().strftime('%H:%M:%S'))
        model, model_params, stats = train_model(train_data, model, model_params, trainer, dev_data)
        train_acc_list_i, train_loss_list_i, dev_acc_list_i, dev_loss_list_i = stats
        train_acc_list += train_acc_list_i
        train_loss_list += train_loss_list_i
        dev_acc_list += dev_acc_list_i
        dev_loss_list += dev_loss_list_i

    test_data = test_src_data, test_target_data, test_label_data
    print("Test started at: " + datetime.datetime.now().strftime('%H:%M:%S'))
    test_acc, test_loss = predict_test(test_data, model, model_params, trainer)
    print("==== TEST loss=%.4f acc=%.2f%% ====" % (test_loss, test_acc))

    make_statistics(train_loss_list, train_acc_list, "train")
    make_statistics(dev_loss_list, dev_acc_list, "dev")
    with open('test_stat.txt', 'w') as f:
        f.write("acc: %F\nloss: %f" % (test_acc, test_loss))

    print("Finished at: " + datetime.datetime.now().strftime('%H:%M:%S'))

