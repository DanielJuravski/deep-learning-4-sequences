import sys

LINES_START = 0
LINES_LIMIT = float('inf')

def compare_predictions_to_answers(predictions_file, answers_file):
    with open(answers_file) as answers_f:
        with open(predictions_file) as pred_f:
            answers_content=answers_f.readlines()
            answers_content=[x.split() for x in answers_content]
            pred_content=pred_f.readlines()
            pred_content=[x.split() for x in pred_content]
            bad = 0.0
            good = 0.0

            for line_i in range(1, len(answers_content)):
                if answers_content[line_i]:
                    gold_word = answers_content[line_i][0]
                    gold_tag = answers_content[line_i][1]
                    pred_word = pred_content[line_i][0]
                    pred_tag = pred_content[line_i][1]

                    if gold_word != pred_word:
                        print "different words"
                        raise ()

                    if gold_tag == pred_tag:
                        good += 1
                    else:
                        bad += 1



    return (good, bad)


if __name__ == '__main__':
    our_output_file = sys.argv[1]
    tagged_file = sys.argv[2]

    (good_results, bad_results) = compare_predictions_to_answers(our_output_file, tagged_file)
    print "good results = " + repr(good_results)
    print "bad results = " + repr(bad_results)
    print "precentage = " + repr(float(good_results) * 100 / (good_results + bad_results))
