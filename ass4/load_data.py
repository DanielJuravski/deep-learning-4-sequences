import json
import numpy as np
import datetime


ANNOTATOR_DICT = {}
ANNOTATOR_DICT['neutral'] = 0
ANNOTATOR_DICT['contradiction'] = 1
ANNOTATOR_DICT['entailment'] = 2

NUM_OF_OOV_EMBEDDINGS = 100
LEN_EMB_VECTOR = 300
OOV_EMBEDDING_STR = 'OOV'

def loadSNLI_labeled_data(snli_file, data_type='not train'):
    print "File " + snli_file + " started loading at: " + datetime.datetime.now().strftime('%H:%M:%S')

    sen1_data = []
    sen2_data = []
    label_data = []

    with open(snli_file) as f:
        f_lines = f.readlines()
        if data_type == 'train':
            smaple_range = 1501
        else:
            smaple_range = len(f_lines)
        for line_i in range(smaple_range):
            line = f_lines[line_i]
            line_json_data = json.loads(line)
            annotator_str_label = str(line_json_data[u'annotator_labels'][0])
            annotator_label = ANNOTATOR_DICT[annotator_str_label]
            if annotator_str_label != '-':
                sen1 = str(line_json_data[u'sentence1'])
                sen2 = str(line_json_data[u'sentence2'])
                sen1_data.append(sen1)
                sen2_data.append(sen2)
                label_data.append(annotator_label)
    print "File " + snli_file + " done loading at: " + datetime.datetime.now().strftime('%H:%M:%S')

    return sen1_data, sen2_data, label_data


def get_emb_data(glove_emb_file):
    emb_dict = {}
    dict_i = 0
    emb = []
    print "File " + glove_emb_file + " started loading at: " + datetime.datetime.now().strftime('%H:%M:%S')

    with open(glove_emb_file) as f:
        f_lines = f.readlines()
        for line_i in range(len(f_lines)):
        #for line_i in range(1000):
            line = f_lines[line_i]
            line_arr = line.split()
            word_str = line_arr[0]
            word_vec = np.array([(float(x)) for x in line_arr[1:]])
            emb.append(word_vec)
            emb_dict[word_str] = dict_i
            dict_i+=1
    # add to dict emb for oov words
    eps = np.sqrt(6) / np.sqrt(LEN_EMB_VECTOR + NUM_OF_OOV_EMBEDDINGS)
    for i in range(NUM_OF_OOV_EMBEDDINGS):
        word_str = OOV_EMBEDDING_STR + str(i)
        emb_vec = np.random.uniform(-eps, eps, LEN_EMB_VECTOR)
        emb.append(emb_vec)
        emb_dict[word_str] = dict_i
        dict_i += 1

    print "File " + glove_emb_file + " done loading at: " + datetime.datetime.now().strftime('%H:%M:%S')

    return emb_dict, emb

