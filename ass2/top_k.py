import sys
import numpy as np

OUTPUT_FILE = "closestWordsOut.txt"
vocab_dict = {}

def load_wordVectorsVocab(vocab_data, wordVectors_data):
    vocab_vector = {}

    with open(vocab_data) as vocab_f, open(wordVectors_data) as wordVector_f:
        vocab_f_num_lines = sum(1 for line in vocab_f)
        wordVector_f_num_lines = sum(1 for line in wordVector_f)
        if vocab_f_num_lines != wordVector_f_num_lines:
            print "Number of lines in %s and %s is not identical !!!" % (vocab_data, wordVectors_data)
            raise AssertionError()
        # define file pointer to the head of the file
        vocab_f.seek(0)
        wordVector_f.seek(0)

        for word, vector_str in zip(vocab_f, wordVector_f):
            vector = [float(x) for x in vector_str.split()]
            vocab_vector[word.strip()] = vector

    print "Word to vector dictionary was loaded."
    return vocab_vector


def print_results(file, word, closest_words):
    file.write("Word is : %s\n" % word)
    file.write("closest words are:\n")
    for w in closest_words:
        file.write("  \"%s\"\twith dist of:%s\n" % (w[0],w[1]))
    file.write("\n******************\n")


def dist_from_word(word_1, word_2):
    u = vocab_dict[word_1]
    v = vocab_dict[word_2]
    domminator = np.dot(u,v)
    denominator = np.sqrt(np.dot(u,u)) * np.sqrt(np.dot(v,v))
    return domminator/denominator


def most_similar(word, k):
    word_dist_arr = []
    for w in vocab_dict.keys():
        if w != word:
            word_dist_arr.append((w, dist_from_word(word, w)))

    sorted_tuples = sorted(word_dist_arr, key=lambda tup: -tup[1])[:k]
    #return [tup[0] for tup in sorted_tuples]
    return sorted_tuples


def print_k_closest_words(k, words):
    output_f = open(OUTPUT_FILE, 'w')
    for word in words:
        closest_words = most_similar(word, k)
        print_results(output_f, word, closest_words)
    output_f.close()

if __name__ == '__main__':
    
    if len(sys.argv) > 3:
        vocab_file = sys.argv[1]
        vectors_file = sys.argv[2]
        k = int(sys.argv[3])
    else:
	vocab_file = "data/vocab.txt"
    	vectors_file ="data/wordVectors.txt"
    	k = 5

    vocab_dict = load_wordVectorsVocab(vocab_file, vectors_file)
    words_to_check = ["dog", "england", "john", "explode", "office"]

    print_k_closest_words(k, words_to_check)
