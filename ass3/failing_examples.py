import sys
import random


def create_repeated_seq(half_length):
    chars = "123456789abcd"
    part1 = "".join([random.choice(chars) for x in range(half_length)])
    part2 = part1
    valid_seq = part1 + part2

    change_index = random.randint(0, half_length-1)
    new_l = random.choice(chars)
    while new_l == part2[change_index]:
        new_l = random.choice(chars)
    part3_l = list(part2)
    part3_l[change_index] = new_l
    part3 = "".join(part3_l)
    invalid_seq = part1 + part3

    return valid_seq, invalid_seq


def create_seq_with_char_at_index(length, index_by_ones=True, at_end=True):
    letters = "abcd"
    part1 = "".join([random.choice(letters) for x in range(length)])
    index = random.randint(0, length-1)
    if index_by_ones:
        index_str = "".join(['1']*(index+1))
    else:
        index_str=str(index)


    valid_part2 = part1[index] + index_str

    new_l = random.choice(letters)
    while new_l == part1[index]:
        new_l = random.choice(letters)
    p1l = list(part1)
    p1l[index] = new_l
    invalid_part1 = "".join(p1l)

    if at_end:
        return part1+valid_part2, invalid_part1+valid_part2
    else:
        return valid_part2+part1, valid_part2+invalid_part1


def create_seq_primes(primes):
    p = random.sample(primes, 1)[0]
    non_p = random.randint(2, 15485863)
    is_easy = non_p % 2 ==0 or non_p % 5 == 0
    while non_p in primes or is_easy:
        non_p = random.randint(2, 15485863)
        is_easy = non_p % 2 ==0 or non_p % 5 == 0

    return str(p), str(non_p)


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



def load_primes():
    primes_file = open("data/fail_primes/primes1.txt")
    content = primes_file.readlines()
    primes = set()
    for i in content:
        for num in i.split():
            primes.add(int(num))

    return primes

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

    if "prime" in type:
        primes = load_primes()

    pos_example_list = []
    neg_example_list = []

    for i in range(num_of_examples):
        if "repeat" in type:
            length = random.randint(min_seq_len, max_seq_len)
            pos, neg = create_repeated_seq(length)

        elif "index" in type:
            length = random.randint(min_seq_len, max_seq_len)
            pos, neg = create_seq_with_char_at_index(length)
        elif "prime" in type:
            pos, neg = create_seq_primes(primes)
        else:
            print("unkown type requested")
            exit(1)

        neg_example_list.append(neg)
        pos_example_list.append(pos)

    write2files(pos_example_list, neg_example_list, type)

