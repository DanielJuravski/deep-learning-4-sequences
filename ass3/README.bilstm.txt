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
	--analyse: For getting dev accuracy data saved and show plot. (available only if --dev-file is turned). (optional).
	--vocab <vocab file>: For setting your own vocab. Default vocab is the one of Ass2. (works only if --wordVectors is turned). (optional).
	--wordVectors <wordVectors file>: For setting your own embedding. Default embedding is the one of Ass2. (works only if --vocab is turned). (optional).



############################
### Running bilstmTag.py ###
############################	

$ python bilstmTag.py <repr> <modelFile> <inputFile> [options]

-options:
	--output <pred file>: Useful for not overwrite the exist data, but write the preds to a new file. If not user, default is the <inputFile>. (optional).

That script assumes that, the <modelFile> that is passed, fitts the <repr> that is passed. Otherwise there will be an exeption.
i.e:
$ python bilstmTrain.py a trainFile modelFile
will create model that can feed only the next command:
$ python bilstmTag.py a modelFile inputFile
and so on...
