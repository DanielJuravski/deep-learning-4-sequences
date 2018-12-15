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



############################
### Running bilstmTag.py ###
############################	

$ python bilstmTag.py <repr> <modelFile> <inputFile> [options]

-options:
	--output <pred file>: Path for the predicted data output. If not given, default is the test.pred in current directory. (optional).

That script assumes that, the <modelFile> that is passed, fitts the <repr> that is passed. Otherwise there will be an exeption.
i.e:
$ python bilstmTrain.py a trainFile modelFile
will create model that can feed only the next command:
$ python bilstmTag.py a modelFile inputFile
and so on...
