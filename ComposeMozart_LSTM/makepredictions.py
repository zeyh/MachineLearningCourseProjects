'''
Zeyang Huang
CSC420 - final project
Due May 12, 2018

part - 3
~ need the file
~ "output.txt", "weights-improvement-04-3.4882.hdf5", "model_regularizer.yaml" to run

load the model and choose a weight to load
generate predictions and print them

tutorial: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
reference: https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/
'''


import numpy as np
from random import randint
from keras.models import model_from_yaml

sequencelength = 80 #sequence length!! need to match with model's shape!

'''reload the model and predict'''
def reload(inputfile, weightfilename):
    # Get a unique identifier for each char in the inpt file,
    # then make some dicts to ease encoding and decoding
    chars = sorted(list(set(inputfile)))
    encoding = {c: i for i, c in enumerate(chars)}
    decoding = {i: c for i, c in enumerate(chars)}

    numberofchars = len(chars) #some parameters from lstmtrain.py
    filelength = len(inputfile)
    skip = 1

    #convert the characters to integers using lookup table 
    dataX = []
    dataY = []
    for i in range(0, len(inputfile) - sequencelength, skip):
        seq_in = inputfile[i:i + sequencelength]
        seq_out = inputfile[i + sequencelength]
        dataX.append([encoding[char] for char in seq_in])
        dataY.append(encoding[seq_out])
        
    n_patterns = len(dataX) #number of sentences
    print("Sliced our file into {0} sentences of length {1}".format(n_patterns, sequencelength))

    
    #build the network from loaded architecture and weights
    #open the model --------- different name for different approaches!!:
    #"model_regularizer.yaml" - model with large penalty
    #"model_regularizer_v2.yaml" - model with small penalty
    #"model.yaml" - no penalty
    #model needs to match the weights h5py file
    with open("model_regularizer_v2.yaml") as model_file:
        architecture = model_file.read()
    model = model_from_yaml(architecture)

    #load weights
    model.load_weights(weightfilename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')


    #a fixed seed
    #generate the seed phrase from the data pattern
##    seed_phrase=""
##    if seed_phrase:
##        phrase_length = len(seed_phrase)
##        seed_pattern = ""
##        for i in range (0, sequencelength):
##            seed_pattern += seed_phrase[i % phrase_length]
##    else:
##        seed = randint(0, len(inputfile) - sequencelength)
##        seed_pattern = inputfile[seed:seed + sequencelength]
##    print("seed pattern: ")
##    print(seed_pattern)

    X = makeseedpattern(inputfile, numberofchars, encoding)
        
    generated_text = ""
        
    checklistDuplicate =[]
    sublist = []
    prevpred = []
    count = 0
    print("predict: ")
    for i in range(500): #number of prediction       
        prediction = np.argmax(model.predict(X, verbose=0)) #pick the largest one as prediction
        prevpred.append(prediction)
        if prediction != 4:
            sublist = []
            sublist.append(prediction)
        else:
            checklistDuplicate.append(sublist)
            for i in range(len(checklistDuplicate)-1):
                if checklistDuplicate[i] == checklistDuplicate[i+1]:
##                    print(count)
                    count = count + 1
                    #force it to change chords
                    if count > 6:
                        X = makeseedpattern(inputfile, numberofchars, encoding)  
                                 
        
        generated_text += decoding[prediction]
##        print(generated_text)
        activations = np.zeros((1, 1, numberofchars), dtype=np.bool)
        activations[0, 0, prediction] = 1
##        print(activations)
        X = np.concatenate((X[:, 1:, :], activations), axis=1)

 
##    print(checklistDuplicate)
    print(generated_text)

'''generate the seed patten for prediction'''
def makeseedpattern(inputfile, numberofchars,encoding):
        #a fixed seed
    #generate the seed phrase from the data pattern
    seed_phrase=""
    if seed_phrase:
        phrase_length = len(seed_phrase)
        seed_pattern = ""
        for i in range (0, sequencelength):
            seed_pattern += seed_phrase[i % phrase_length]
    else:
        seed = randint(0, len(inputfile) - sequencelength)
        seed_pattern = inputfile[seed:seed + sequencelength]

##    seed_pattern = "@H0 @H0 @H+ @H$ @H0 @H0 @I9 @I9 AJ7 AJ5 AJ5 BK; BK; DL9 DL(4 @(4 @(4 HL&2 HL$0 @$0 @$0 IL!- IL%1"
    #generate the predictions!! and match the chars
    X = np.zeros((1, sequencelength, numberofchars), dtype=np.bool)
    #Return a new array of given shape and type, filled with zeros
    for i, character in enumerate(seed_pattern):
        X[0, i, encoding[character]] = 1
        
    return X



def main():
    #"model_regularizer.yaml"  -"weights-improvement-04-3.4882.hdf5"
    #"model_regularizer_v2_justmelody.yaml' - "weights-03-3.0726.hdf5"
    infile = open("output.txt","r")
    lines = infile.read()
    reload(lines, "weights-improvement-04-3.4882.hdf5")
    
main()
