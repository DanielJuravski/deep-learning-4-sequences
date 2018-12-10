import random
import sys


def get_train_and_validation(train_pos_file, train_neg_file):

    all_x =[]
    all_y = []
    with open(train_pos_file) as f:
        for line in f.readlines():
            all_x.append(line)
            all_y.append(1)

    with open(train_neg_file) as f:
        for line in f.readlines():
            all_x.append(line)
            all_y.append(0)

    combined = list(zip(all_x, all_y))
    random.shuffle(combined)
    all_x, all_y = zip(*combined)
    n = int(len(all_y) * 0.8)
    return all_x[:n], all_y[:n], all_x[n:], all_y[n:]


def init_model():
    pass


def train(model, tagged_samples):
    train_x, train_y = tagged_samples


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
    model = init_model()
    trained_model = train(model, (train_x, train_y))