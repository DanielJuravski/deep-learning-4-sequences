import random
import sys
import dynet as dy
import numpy as np

EPOCHS = 10

MLP_OUTPUT_SIZE = 2

N1 = 22

LAYERS = 1
INPUT_DIM = 20
HIDDEN_DIM = 20
LSTM_OUTPUT_SIZE = 18


int2char = None
char2int = None
def get_train_and_validation(train_pos_file, train_neg_file):

    all_x =[]
    all_y = []
    with open(train_pos_file) as f:
        for line in f.readlines():
            all_x.append(line.strip())
            all_y.append(1)

    with open(train_neg_file) as f:
        for line in f.readlines():
            all_x.append(line.strip())
            all_y.append(0)

    all_x, all_y = shuffle(all_x, all_y)
    n = int(len(all_y) * 0.8)
    return all_x[:n], all_y[:n], all_x[n:], all_y[n:]


def shuffle(x, y):
    combined = list(zip(x, y))
    random.shuffle(combined)
    x, y = zip(*combined)
    return x, y


def init_model():
    characters = list("123456789abcd ")
    characters.append("<EOS>")
    global int2char
    int2char = list(characters)
    global char2int
    char2int = {c:i for i,c in enumerate(characters)}
    global VOCAB_SIZE
    VOCAB_SIZE = len(characters)

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
    # loss = []
    for char in sentence:
        s = s.add_input(lookup[char])

    lstm_out = (R * s.output()) + bias
    yhat = dy.softmax(mlp(lstm_out, params))
    loss = -(dy.log(dy.pick(yhat, y)))

    #loss = dy.esum(loss)
    return loss, yhat


def train(model, tagged_samples, trainer):
    train_x, train_y = tagged_samples
    lstm, params, pc = model

    correct = 0
    for i in range(len(train_y)):
        line = train_x[i]
        y = train_y[i]
        loss, yhat = predict(lstm, params, line, y)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 100 == 0:
            print("loss value= %.10f" % loss_value )
          #  print("prediction= " + repr(yhat.npvalue()))

        prediction = np.argmax(yhat.npvalue())
        if prediction == train_y[i]:
            correct += 1

    precent = correct/float(len(train_y))
    print("percent = %f" % precent)


def validate(model, dev_samples):
    dev_x, dev_y = dev_samples
    lstm, params, pc = model

    correct = 0
    for i in range(len(dev_y)):
        line = dev_x[i]
        y = dev_y[i]
        loss, yhat = predict(lstm, params, line, y)
        loss_value = loss.value()
        if i % 100 == 0:
            print("validation loss value= %.10f" % loss_value )

        prediction = np.argmax(yhat.npvalue())
        if prediction == y:
            correct += 1

    precent = correct/float(len(dev_y))
    print("validation percent = %f" % precent)


if __name__ == '__main__':
    train_pos_file = "data/pos_examples"
    train_neg_file = "data/neg_examples"
    test_pos_file = "data/pos_test"
    test_neg_file = "data/neg_test"

    if len(sys.argv) > 3:
        train_pos_file = sys.argv[1]
        train_neg_file = sys.argv[2]
        test_pos_file = sys.argv[3]
        test_neg_file = sys.argv[4]

    train_x, train_y, dev_x, dev_y = get_train_and_validation(train_pos_file, train_neg_file)
    model, trainer = init_model()

    for i in range(EPOCHS):
        print ("epoch no: %d" % i)
        train_x, train_y = shuffle(train_x, train_y)
        trained_model = train(model, (train_x, train_y), trainer)

        dev_x, dev_y = shuffle(dev_x, dev_y)
        validate(model, (dev_x, dev_y))