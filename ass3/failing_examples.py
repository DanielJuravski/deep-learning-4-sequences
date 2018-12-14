import sys
import random


'''
1) repeated sequence, this will require big enough memory to represent at least half the size of the sequence
and therefore we can increase the sequence. if the sequence is bigger than 2^(|output_vector| + 1 ) than it cannot be
remembered with order

2) index of reoccurrence to last letter.
each sequence of letters and digits ends with a number smaller than the sequence length, and a letter to the left of it
if that letter appears in the sequence at the index signified by the number than the sequence is in the language,
otherwise it isnt:
example
avfadfjkgadf2
since sequence[2] == f this is a valid example
We assume that due to the memory limitations its hard to solve while the other direction would have been easier

3) prime number detection

'''

chars = "123456789abcd"

# def createSeqWithIndexToReaccuringLetter(valid):
#     if valid:

def create_repeated_seq(half_length, is_valid):

    part1 = "".join([random.choice(chars) for x in range(half_length)])
    part2 = part1
    if not is_valid:
        change_index = random.randint(0, half_length-1)
        new_l = random.choice(chars)
        while new_l == part2[change_index]:
            new_l = random.choice(chars)
        part2_l = list(part2)
        part2_l[change_index] = new_l
        part2 = "".join(part2_l)

    return part1+part2


def create_seq_with_char_at_index(length, is_valid, at_end=True):
    part1 = "".join([random.choice(chars) for x in range(length)])
    index = random.randint(0, length-1)
    if is_valid:
        part2 = part1[index] + str(index)
    else:
        new_l = random.choice(chars)
        while new_l == part1[index]:
            new_l = random.choice(chars)
        part2 = new_l + str(index)

    if at_end:
        return part1+part2
    else:
        return part2+part1



def write2files(pos_example_list, neg_example_list, type):
    pos ='pos_examples_' + type
    with open(pos, 'w') as f:
        for example in pos_example_list:
            f.write(example)
            f.write("\n")

    neg = 'neg_examples_' + type
    with open(neg, 'w') as f:
        for example in neg_example_list:
            f.write(example)
            f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        type = sys.argv[1]
    else:
        type = "index"

    if len(sys.argv) > 3:
        min_seq_len = int(sys.argv[2])
        max_seq_len = int(sys.argv[3])
        num_of_examples = int(sys.argv[4])  # each type
    else:
        min_seq_len = 10
        max_seq_len = 20
        num_of_examples = 500

    pos_example_list = []
    for i in range(num_of_examples):
        if "repeat" in type:
            length = random.randint(min_seq_len, max_seq_len)
            x = create_repeated_seq(length, True)
        elif "index" in type:
            length = random.randint(min_seq_len, max_seq_len)
            x = create_seq_with_char_at_index(length, True)
        pos_example_list.append(x)

    neg_example_list = []
    for i in range(num_of_examples):
        if "repeat" in type:
            length = random.randint(min_seq_len, max_seq_len)
            x = create_repeated_seq(length, False)
        elif "index" in type:
            length = random.randint(min_seq_len, max_seq_len)
            x = create_seq_with_char_at_index(length, True)
        neg_example_list.append(x)

    write2files(pos_example_list, neg_example_list, type)

