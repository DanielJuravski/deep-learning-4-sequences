import sys
import random


def makeExamples(alphas, min_digit_seq, max_digit_seq, num_of_examples):
    examples = []
    for example_i in range(num_of_examples):
        example = "".join([random.choice("123456789") for x in range(random.randint(min_digit_seq, max_digit_seq))])
        for alpha in alphas:
            example += "".join([random.choice(alpha) for x in range(random.randint(min_digit_seq, max_digit_seq))])
            example += "".join([random.choice("123456789") for x in range(random.randint(min_digit_seq, max_digit_seq))])
        examples.append(example)

    return examples


def write2files(pos_example_list, neg_example_list):
    with open('pos_examples', 'w') as f:
        for example in pos_example_list:
            f.write(example)
            f.write("\n")

    with open('neg_examples', 'w') as f:
        for example in neg_example_list:
            f.write(example)
            f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) > 3:
        min_digit_seq = int(sys.argv[1])
        max_digit_seq = int(sys.argv[2])
        num_of_examples = int(sys.argv[3])  # each type
    else:
        min_digit_seq = 10
        max_digit_seq = 20
        num_of_examples = 500

    pos_example_list = makeExamples(alphas=('a', 'b', 'c', 'd'),
                                    min_digit_seq=min_digit_seq,
                                    max_digit_seq=max_digit_seq,
                                    num_of_examples=num_of_examples)
    neg_example_list = makeExamples(alphas=('a', 'c', 'b', 'd'),
                                    min_digit_seq=min_digit_seq,
                                    max_digit_seq=max_digit_seq,
                                    num_of_examples=num_of_examples)

    write2files(pos_example_list, neg_example_list)

