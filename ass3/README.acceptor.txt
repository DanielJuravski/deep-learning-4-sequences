
######################
packages:
dynet
numpy

######################
how to run:
python expirement.py <positive_examples_file> <negative_examples_file> <positive_examples_test_file> <negative_examples_test_file> [<expirement_name> = basic]

Where the first 4 parameters are positive/negative examples for train and test.
The fifth paramater is optional and is the name pof the experiment which will be prefixed to the outputs

#####################
Output:
The program prints to the console the average loss and accuracy for each epoch on train and validation data,
where validation data is 20 percent of the train data.
The program also prints the accuracy percentage on the test data.

The program saves to the current folder .npy files as so:
array of train loss per epoch - basic_train_loss.npy
array of train accuracy per epoch - basic_train_acc.npy
array of validation loss per epoch - basic_dev_loss.npy
array of validation accuracy per epoch - basic_dev_acc.npy
array of size 1 holding test accuracy = basic_test_acc.npy

If a fifth parameter was supplied to the program the names of the file will be prefixed by that string
instead of "basic_"


