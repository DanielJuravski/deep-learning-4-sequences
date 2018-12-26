import json
import numpy as np
import datetime
import fileinput


ANNOTATOR_DICT = {}
ANNOTATOR_DICT['neutral'] = 0
ANNOTATOR_DICT['contradiction'] = 1
ANNOTATOR_DICT['entailment'] = 2

NUM_OF_OOV_EMBEDDINGS = 100
LEN_EMB_VECTOR = 300
OOV_EMBEDDING_STR = 'OOV'

def loadSNLI_labeled_data(snli_file):
    """
    load snli labeled data from file, filter '-' annotators
    :param snli_file: train or dev files
    :return: array of tuples of data, each array var is a tuple of (sen1[str], sen2[str], label[int])
    """
    data = []
    with open(snli_file) as f:
        f_lines = f.readlines()
        #for line_i in range(len(f_lines)):
        for line_i in range(10000):
            line = f_lines[line_i]
            line_json_data = json.loads(line)
            annotator_str_label = str(line_json_data[u'annotator_labels'][0])
            annotator_label = ANNOTATOR_DICT[annotator_str_label]
            if annotator_str_label != '-':
                sen1 = str(line_json_data[u'sentence1'])
                sen2 = str(line_json_data[u'sentence2'])
                data.append((sen1, sen2, annotator_label))

    print "File " + snli_file + " was loaded at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return data


def loadSNLI_unlabeled_data(snli_test_file):
    """
    load snli unlabeled data from file
    :param snli_file: test file
    :return: array of tuples of data, each array var is a tuple of (sen1[str], sen2[str])
    """
    test_data = []
    with open(snli_test_file) as f:
        f_lines = f.readlines()
        for line in f_lines:
            line_json_data = json.loads(line)
            sen1 = str(line_json_data[u'sentence1'])
            sen2 = str(line_json_data[u'sentence2'])
            test_data.append((sen1, sen2))

    return test_data


def get_emb_data(glove_emb_file):
    """
    get emb dict
    :param glove_emb_file:
    :return: dict where key is word, val is [300,1] ndarray word embedding
    """
    emb_dict = {}
    with open(glove_emb_file) as f:
        f_lines = f.readlines()
        for line_i in range(len(f_lines)):
        #for line_i in range(1000):
            line = f_lines[line_i]
            line_arr = line.split()
            word_str = line_arr[0]
            word_vec = np.array([(float(x)) for x in line_arr[1:]])#.reshape(1,LEN_EMB_VECTOR)
            emb_dict[word_str] = word_vec
    # add to dict emb for oov words
    eps = np.sqrt(6) / np.sqrt(LEN_EMB_VECTOR + NUM_OF_OOV_EMBEDDINGS)
    for i in range(NUM_OF_OOV_EMBEDDINGS):
        word_str = OOV_EMBEDDING_STR + str(i)
        emb_vec = np.random.uniform(-eps, eps, LEN_EMB_VECTOR)#.reshape(1,LEN_EMB_VECTOR)
        emb_dict[word_str] = emb_vec

    print "File " + glove_emb_file + " was loaded at: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return emb_dict


if __name__ == '__main__':
    # snli_train_file = 'data/snli_1.0/snli_1.0_train.jsonl'
    # snli_dev_file = 'data/snli_1.0/snli_1.0_dev.jsonl'
    # snli_test_file = 'data/snli_1.0/snli_1.0_test.jsonl'

    # train_data = loadSNLI_labeled_data(snli_train_file)
    # dev_data = loadSNLI_labeled_data(snli_dev_file)
    # test_data = loadSNLI_unlabeled_data(snli_test_file)

    glove_emb_file = 'data/glove/glove.6B.300d.txt'
    emb_data = get_emb_data(glove_emb_file)

    pass