'''
Due May 12, 2018

part - 2
~ need the file "output.txt" to run

take the txt file with ascii char patterns
load the network and train the model
model saved in "model.yaml"
weights saved in weights-improvement

tutorial: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
reference: https://keras.io/layers/recurrent/#lstm
'''
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
import functools
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import History
from keras.regularizers import l2,l1


#ignore the error: keep_dims is deprecated, use keepdims insteÍad
tf.logging.set_verbosity(tf.logging.ERROR) 


def main():
    infile = open("output.txt","r")
    lines = infile.read()
    print(type(lines))
    print("load a file of {0} characters".format(len(lines)))
    lstmtest(lines)


''' take the file as input and train the LSTM neural network!'''
def lstmtest(allpiecelist):
##    allpiecelist = ''.join(allpiecelist) #turn into a char list without space   
##    allpiecelist = removeSpaces(allpiecelist) #turn into a char list without space
##    print(allpiecelist[1:200])
    notelist = list(allpiecelist)
    print("length of the char: ",len(notelist))
    #fix the random number seed to ensure the results are reproducible.
    np.random.seed(7)
    
##    #change the 2d list to np array and flatten into 1d
##    allpiecelist = flatten(allpiecelist)
##    char = sort_and_deduplicate(allpiecelist)
    
    #normalize the dataset
    #create mapping of unique chars to integers
    sortedlist = sorted(list(set(notelist)))
##    sortedlist.remove(" ")
##    print(sortedlist)
##    chars = sorted(list(set(notelist)))
    chars = sortedlist
    print("chars:" ,chars) #total piano note range
    encoding = {c: i for i, c in enumerate(chars)}
    decoding = {i: c for i, c in enumerate(chars)}
##    char_to_int = dict((c, i) for i, c in enumerate(chars))
    n_chars = len(notelist)
    n_vocab = len(chars)
    print("Our file contains {0} unique characters.".format(len(chars)))
##    print ("Total Characters: ", n_chars) #overall notelist size
##    print ("Total Vocab: ", n_vocab) #all the indentitical notes


    #convert the characters to integers using lookup table 
    seq_length =150 #sentence length
    skip = 1 # -----?
    dataX = []
    dataY = []
    #chop the data by sequence length
    for i in range(0, len(allpiecelist) - seq_length, skip):
        seq_in = notelist[i:i + seq_length]
        seq_out = notelist[i + seq_length]      
        dataX.append([encoding[char] for char in seq_in])       
        dataY.append(encoding[seq_out])
       
    n_patterns = len(dataX) #number of sequences
    print("Sliced the input file into {0} sequenes of length {1}".format(n_patterns, seq_length))


    #vectorize the x and y:
       
##    #approach 1 # reshape X to be [samples, time steps, features]
##    X = np.reshape(dataX, (n_patterns, seq_length, 1))
##    # normalize
##    X = X / float(n_vocab)
##    # one hot encode the output variable
##    y = np_utils.to_categorical(dataY)
##
    #approach 2 
    print("len",len(chars))
    print("vectorize x y ...")
    #slice the data into x and y with arbitrary overlapping sequences
    #the len of slice of the data, the predefined length, the total num of chars
    X = np.zeros((len(dataX), seq_length, len(chars)) ) #, dtype=np.bool) #filled with boolean falses
    y = np.zeros((len(dataX), len(chars)) ) #, dtype=np.bool)
    #fill the np array
    for i, sequence in enumerate(dataX):
        for t, encoded_char in enumerate(sequence):
            X[i, t, encoded_char] = 1
        y[i, dataY[i]] = 1
        

    #double check the vectorized data before training
    print("Sanity check y. Dimension: {0} # Sentences: {1} Characters in file: {2}".format(y.shape, n_patterns, len(chars)))
    print("Sanity check X. Dimension: {0} Sentence length: {1}".format(X.shape, seq_length))
##    print(X[1])

    
    #define the lstm model
    #return either sequences (return_sequences=True) or a matrix.
    model = Sequential()
    model.add(LSTM(512, input_shape=(seq_length, len(chars)) )) #returning sequence
    model.add(Dropout(0.5)) #probaility

    #add different layers
##    model.add(LSTM(256/len(chars), input_shape=(X.shape)))
##    model.add(Dropout(0.3))
##    model.add(LSTM(256, input_shape=(seq_length, len(chars))))

    #add the regularizers
    model.add(Dense(len(chars), activation='softmax',  kernel_regularizer=keras.regularizers.l2(0.010)))
    model.compile(loss='categorical_crossentropy', optimizer='adam')


    #output the architecture in a different file so that it can be loaded seperately
    architecture = model.to_yaml()
    with open('model_regularizer_v2.yaml', 'a') as model_file:
        model_file.write(architecture)
    
    #define the checkpoint
    filepath="weights-{epoch:02d}-{loss:.4f}.hdf5" #verbose = 1 ? 0
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    #fittttt the model!!!!
    history = History() #just for debugging
    epochsize = 2
    history = model.fit(X, y, epochs=epochsize, batch_size=254, callbacks=callbacks_list)

#for debugging
##    layer = Dense(Dense(y.shape[1], activation='softmax'))  
##    print("layer: ", layer.get_weights())
##    print("history: ",history.history.keys())
##    print("loss: ",history.history['loss'])
    lossnum = history.history['loss'][0]
    lossnum = "%.4f" % lossnum
    epochsize = "{:02d}".format(epochsize)
   

 
#just for reference - just single notes melody
#trannsform ascii char to number that represent a note's pitch
def asciitonum():
##    stringinput = "7747774777007744774477007"
    stringinput = "HGGGHHHHHHHHHHGGGGGGGHHHHHHHGGGGGGGGGGEGGEGGGGGEEEE>G>E>C>@<<<<<<<<<<<<>>>>>><<<<<<<<<<<<7777777++++++++0000000///////////000"
    newnumlist = []
    for i in stringinput:
##        print(ord(i))
        newnumlist.append(ord(i))
    print (newnumlist)
    return newnumlist


# Function to remove all spaces from a given string
def removeSpaces(string):
    # To keep track of non-space character count
    count = 0 
    list = [] 
    # Traverse the given string. If current character
    # is not space, then place it at index 'count++'
    for i in range(len(string)):
        if string[i] != ' ':
            list.append(string[i])
 
    return toString(list)
 
# Utility Function
def toString(List):
    return ''.join(List)

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

#flatten a 2d list
def flatten(l):
    try:
        return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    except IndexError:
        return []

def cls():
    print ("\n" * 100)
    
main()


