import json

ANNOTATOR_DICT = {}
ANNOTATOR_DICT['neutral'] = 0
ANNOTATOR_DICT['contradiction'] = 1
ANNOTATOR_DICT['entailment'] = 2


def loadSNLI_labeled_data(snli_file):
    """
    load snli labeled data from file, filter '-' annotators
    :param snli_file: train or dev files
    :return: array of tuples of data, each array var is a tuple of (sen1[str], sen2[str], label[int])
    """
    data = []
    with open(snli_file) as f:
        f_lines = f.readlines()
        for line in f_lines:
            line_json_data = json.loads(line)
            annotator_str_label = str(line_json_data[u'annotator_labels'][0])
            annotator_label = ANNOTATOR_DICT[annotator_str_label]
            if annotator_str_label != '-':
                sen1 = str(line_json_data[u'sentence1'])
                sen2 = str(line_json_data[u'sentence2'])
                data.append((sen1, sen2, annotator_label))

    return data


def loadSNLI_unlabeled_data(snli_test_file):
    """
    load snli labeled data from file
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


if __name__ == '__main__':
    snli_train_file = 'data/snli_1.0/snli_1.0_train.jsonl'
    snli_dev_file = 'data/snli_1.0/snli_1.0_dev.jsonl'
    snli_test_file = 'data/snli_1.0/snli_1.0_test.jsonl'

    train_data = loadSNLI_labeled_data(snli_train_file)
    dev_data = loadSNLI_labeled_data(snli_dev_file)
    test_data = loadSNLI_unlabeled_data(snli_test_file)

    pass