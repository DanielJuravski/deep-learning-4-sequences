
######################
packages:
dynet
numpy

######################
how to run:
python expirement.py <positive_examples_file> <negative_examples_file> <positive_examples_test_file> <negative_examples_test_file>

Where the first 4 parameters are positive/negative examples for train and test.

#####################
Output:
The program prints to the console the average loss and accuracy for each epoch on train and validation data,
where validation data is 20 percent of the train data.
The program also prints the accuracy percentage on the test data.



