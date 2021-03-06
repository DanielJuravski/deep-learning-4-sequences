python version: 2.7

Required packeges:
	-'random'
	-'dynet'
	-'numpy'
	-'matplotlib.pyplot
	-'pickle'


##############################
### Running bilstmTrain.py ###
##############################

$ python bilstmTrain.py <repr> <trainFile> <modelFile> [options]

-options:
	--dev-file <dev file>: For seeing dev statistics (loss and accuracy) during the learning. (optional).
	--analyse: For getting dev accuracy data saved and show plot. (available only if --dev-file option is given). (optional).
	--vocab <vocab file>: For setting your own vocab. Default vocab is the one of Ass2. (works only if --wordVectors option is given). (optional).
	--wordVectors <wordVectors file>: For setting your own embedding. Default embedding is the one of Ass2. (works only if --vocab option is given). (optional).

bilstmTrain.py outputs 2 files to the current directory (tag_set.pkl, <repr>_vocab.pkl).
These are required for running bilstmTag.py as explained in the next section.

############################
### Running bilstmTag.py ###
############################	

$ python bilstmTag.py <repr> <modelFile> <inputFile> [options]

-options:
	--output <pred file>: Path for the predicted data output. If not given, default is the test.pred in current directory. (optional).
	--vocab <vocab file>: Path for a .pkl file describing the vocabulary, outputed by bilstmTrain.py.
	If not given the default is <repr type>_vocab.pkl in current directory.
	--tags <tag set file>: Path for a .pkl file describing the tag set, outputed by bilstmTrain.py.
    If not given the default is tag_set.pkl in current directory.

That script assumes that:
1. The <modelFile> that is passed, fits the <repr> that is passed. Otherwise there will be an exeption.
i.e:
$ python bilstmTrain.py a trainFile modelFile
will create model that can feed only the next command:
$ python bilstmTag.py a modelFile inputFile
and so on...

