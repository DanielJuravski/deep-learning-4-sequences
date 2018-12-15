import random
import sys

import datetime
import dynet as dy
import numpy as np

EPOCHS = 5

MLP_OUTPUT_SIZE = 2

LAYERS = 1
INPUT_DIM = 16
HIDDEN_DIM = 20
LSTM_OUTPUT_SIZE = 10
N1 = 8

int2char = None
char2int = None

train_loss = []
dev_loss = []
train_acc = []
dev_acc = []
test_acc = 0
def get_train_and_validation(train_pos_file, train_neg_file):

    all_x, all_y = get_xy_from_files(train_neg_file, train_pos_file)

    all_x, all_y = shuffle(all_x, all_y)
    n = int(len(all_y) * 0.8)
    return all_x[:n], all_y[:n], all_x[n:], all_y[n:]


def get_xy_from_files(train_neg_file, train_pos_file):
    all_x = []
    all_y = []
    with open(train_pos_file) as f:
        for line in f.readlines():
            all_x.append(line.strip())
            all_y.append(1)
    with open(train_neg_file) as f:
        for line in f.readlines():
            all_x.append(line.strip())
            all_y.append(0)
    return all_x, all_y


def shuffle(x, y):
    combined = list(zip(x, y))
    random.shuffle(combined)
    x, y = zip(*combined)
    return x, y


def init_model():
    characters = list("0123456789abcd ")
    characters.append("<EOS>")
    global int2char
    int2char = list(characters)
    global char2int
    char2int = {c:i for i,c in enumerate(characters)}
    global VOCAB_SIZE
    VOCAB_SIZE = len(characters)

    dyparams = dy.DynetParams()
    dyparams.set_random_seed(666)
    dyparams.init()

    pc = dy.ParameterCollection()
    lstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
    params = {}
    params["lookup"] = pc.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
    params["R"] = pc.add_parameters((LSTM_OUTPUT_SIZE, HIDDEN_DIM))
    params["bias"] = pc.add_parameters((LSTM_OUTPUT_SIZE))
    params["w1"] = pc.add_parameters((N1, LSTM_OUTPUT_SIZE))
    params["w2"] = pc.add_parameters((MLP_OUTPUT_SIZE, N1))
    params["b1"] = pc.add_parameters((N1))
    params["b2"] = pc.add_parameters((MLP_OUTPUT_SIZE))

    trainer = dy.RMSPropTrainer(pc)
    return (lstm, params, pc), trainer



def mlp(rnn_ouput, params):
    w1 = params["w1"]
    w2 = params["w2"]
    b1 = params["b1"]
    b2 = params["b2"]

    l1 = dy.tanh((w1*rnn_ouput)+b1)
    out = dy.softmax((w2*l1)+b2)
    return out


def predict(lstm, params, line, y):
    dy.renew_cg()
    s0 = lstm.initial_state()
    R = params["R"]
    bias = params["bias"]
    lookup = params["lookup"]
    sentence = ["<EOS>"] + list(line) + ["<EOS>"]
    sentence = [char2int[c] for c in sentence]
    s = s0

    for char in sentence:
        s = s.add_input(lookup[char])

    lstm_out = (R * s.output()) + bias
    yhat = dy.softmax(mlp(lstm_out, params))
    loss = -(dy.log(dy.pick(yhat, y)))

    return loss, yhat


def train(model, tagged_samples, trainer, epoch_i):
    train_x, train_y = tagged_samples
    lstm, params, pc = model

    correct = wrong = 0.0
    epoch_loss = 0
    for i in range(len(train_y)):
        line = train_x[i]
        y = train_y[i]
        loss, yhat = predict(lstm, params, line, y)
        loss_value = loss.value()
        epoch_loss += loss_value
        loss.backward()
        trainer.update()

        prediction = np.argmax(yhat.npvalue())
        if prediction == train_y[i]:
            correct += 1
        else:
            wrong += 1

        if i > 0 and i % 100 == 0:
            precent = correct / (correct+wrong) * 100
            print("Epoch:%d train iteration:%d loss=%.4f acc=%.2f%%" % (epoch_i, i, loss_value, precent))

    epoch_loss /= len(train_y)
    train_loss.append(epoch_loss)
    train_acc.append((correct / (correct+wrong) * 100))


def validate(model, dev_samples, epoch):
    dev_x, dev_y = dev_samples
    lstm, params, pc = model

    correct = wrong = total_loss_val = 0.0
    for i in range(len(dev_y)):
        line = dev_x[i]
        y = dev_y[i]
        loss, yhat = predict(lstm, params, line, y)
        loss_value = loss.value()
        total_loss_val += loss_value

        prediction = np.argmax(yhat.npvalue())
        if prediction == y:
            correct += 1
        else:
            wrong += 1

    precent = correct/(correct+wrong) * 100
    total_loss = total_loss_val/ float(len(dev_y))
    print("=====epoch%d Validation: loss=%.4f acc=%.2f%% =====" % (epoch, total_loss, precent))
    dev_loss.append(total_loss)
    dev_acc.append(precent)


def test(model, test_set):
    test_x, test_y = test_set
    lstm, params, pc = model

    correct = wrong = 0.0
    for i in range(len(dev_y)):
        line = test_x[i]
        y = test_y[i]
        loss, yhat = predict(lstm, params, line, y)
        prediction = np.argmax(yhat.npvalue())
        if prediction == y:
            correct += 1
        else:
            wrong += 1

    precent = correct / (correct + wrong) * 100
    print("\nTest percent = %2f%%" % precent)
    return precent


if __name__ == '__main__':
    train_pos_file = "data/pos_examples"
    train_neg_file = "data/neg_examples"
    test_pos_file = "data/pos_test"
    test_neg_file = "data/neg_test"
    test_type = "basic"

    if len(sys.argv) > 3:
        train_pos_file = sys.argv[1]
        train_neg_file = sys.argv[2]
        test_pos_file = sys.argv[3]
        test_neg_file = sys.argv[4]

    if len(sys.argv) > 4:
        test_type = sys.argv[5]

    train_x, train_y, dev_x, dev_y = get_train_and_validation(train_pos_file, train_neg_file)
    model, trainer = init_model()

    print("started at " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for i in range(EPOCHS):
        train_x, train_y = shuffle(train_x, train_y)
        trained_model = train(model, (train_x, train_y), trainer, i+1)

        dev_x, dev_y = shuffle(dev_x, dev_y)
        validate(model, (dev_x, dev_y), (i+1))

    test_x, test_y = get_xy_from_files(test_neg_file, test_pos_file)
    test_x, test_y = shuffle(test_x, test_y)
    test_acc = test(model, (test_x, test_y))
    print("finished at " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Lines to save data for graphs
    # np.save(test_type+"_train_loss", train_loss)
    # np.save(test_type+"_dev_loss", dev_loss)
    # np.save(test_type+"_train_acc", train_acc)
    # np.save(test_type+"_dev_acc", dev_acc)
    # np.save(test_type+"_test_acc", [test_acc])