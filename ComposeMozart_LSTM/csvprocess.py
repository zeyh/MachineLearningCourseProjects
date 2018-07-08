'''
Zeyang 
Due May 12, 2018

part - 1
~ need 40 csv files named from "EP1".mid to "EP39.mid" to process
~ generate "output.txt" as txt files for lstmtrain.py to run

take 40 csv files as inputs
extract the time velocity channel and pitch column
sort them by time transform into ascii chars and write to txt
'''


import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tensorflow as tf
import h5py
import functools

#ignore the error: keep_dims is deprecated, use keepdims insteÍad
tf.logging.set_verbosity(tf.logging.ERROR) 


''' read the 40 csv files and call the functin in sequence to process each one '''
''' return a list with all notes compressed into a long string for each piece'''
''' also output "output.txt" with all the strings - example output: "C4 G6 ?6 B7 @7 H9 H7 J7 L7 J7"'''
def readinfiles():
    outfile = open("output_test.txt", "w")
    outfile2 = open("justmelody_test.txt", "w") #JUST THE MELODY
    allpiecelist = []
    justmelody = []
    for i in range(21): #number of overall pieces
        print(i) #how many it processed
        fileindex = str(i+1) #since the file count from 1
        inputfile = pd.read_csv("csvEP/EP"+fileindex+".csv", names=["time","channel","note","velocity"])
        #get the overall timescale from one piece
        timescale = mergelist(inputfile)
        #get the right hand notes and left hand notes seperately
        righthand, lefthand, melody = intoascii(inputfile)
        justmelody.append(melody)
        #sort the notes from two hands and merge into one
        sortedstring = twohandssort(righthand, lefthand, timescale)
        #write to file
        outfile.write(sortedstring)
        allpiecelist.append(sortedstring)


    flattenmelody = flatten(justmelody)
    print("flattern melody len",len(flattenmelody))
    print(len(allpiecelist))
    melodystring = []
    for i in range(len(flattenmelody)):
        melodystring.append(str(chr(flattenmelody[i])))
        outfile2.write(str(chr(flattenmelody[i])))
    return allpiecelist  #turn a long list of strings


 
'''take a panda csv file as input extrate the useful columns'''
'''return two lists that represents the notes from right and left hand seperately '''
def intoascii(inputfile):
    melody = []
    righthand = [] #righthand notes
    lefthand = [] #lefthand notes
    notelist_trans = [] #notes in ascii
    noteextract = inputfile.loc[: , "note"] #extract the notes
    #extract the rows those velocity not 0 and right hand
    extract = inputfile.loc[(inputfile["velocity"]!=0) & (inputfile["channel"] == 0),["note","time"]]
    extractnotelist1 = extract["note"].values.tolist()
    melody.append(extractnotelist1)
    extracttimelist1 = extract["time"].values.tolist()
    righthand.append(extractnotelist1)
    righthand.append(extracttimelist1) 
    #extract the rows those velocity not 0 and right hand
    extract2 = inputfile.loc[(inputfile["velocity"]!=0) & (inputfile["channel"] == 1),["note","time"]]
    extractnotelist2 = extract2["note"].values.tolist()
    extracttimelist2 = extract2["time"].values.tolist()
    lefthand.append(extractnotelist2)
    lefthand.append(extracttimelist2)

##    print(melody[0:100])


    return righthand,lefthand, flatten(melody)


'''combine the lefthand and righthand notes' time together'''
''' get a time scale for the whole piece'''
''' example output:  0, 480, 960, 1440, 1920, 2400, 2640, 2880, 3120...'''
#[0] = note [1]
def mergelist(inputfile):
    newnote = []
    righthand,lefthand, melody = intoascii(inputfile) #get the right hand and left hand lists
    timescale = []
    #compress the two lists together zipped note by note into one list
    timescale = functools.reduce(lambda x,y :x+y ,[righthand[1],lefthand[1]])
    #sort the timescale
    timescale = sorted(timescale)
    #if there are duplication then delete it
    timescale1 = checkduplicated(timescale)
##    print(timescale1[0:100])
    return timescale1

'''take a list as input, check if a list has two element the same if true, delete the duplicate element'''
'''output the reduced list'''
def checkduplicated(inputlist):
    newlist = []
    i = 0
    while i < len(inputlist)-1: #check in advance if the element in i+1 is equal to element in i
        if inputlist[i] == inputlist[i+1]:
            newlist.append(inputlist[i])
            i = i+2 #for scenario if list[i] == list[i+1] == list[i+2]
        else:
            newlist.append(inputlist[i])
            i = i+1
    return newlist

'''take the input of two hands' note and map them on the timescale'''
''' '''
def twohandssort(righthand, lefthand, timescale):
##    print(timescale)
    #right hand extract mapping it into the timescale
    newlist = []
    for i in range(len(timescale)):       
        temp = []
        newlist.append(temp)
        for t in range(len(righthand[0])):          
            if timescale[i] == righthand[1][t]:
                temp.append(str(chr(righthand[0][t])))
    #fill the empty time with the next time slot           
    for i in range(len(newlist)-4): #ignore the last one
        if newlist[i] == []:
            newindex = 0          
            while newlist[i+newindex] == [] and (i != len(newlist)-1):
                newindex = newindex + 1

            newlist[i] = newlist[i+newindex]

    #left hand extract mapping it into the timescale
    newlist1 = []
    for i in range(len(timescale)):       
        temp = []
        newlist1.append(temp)  
        for t1 in range(len(lefthand[0])):
            if timescale[i] == lefthand[1][t1]:
                temp.append(str(chr(lefthand[0][t1])))
    #fill the empty time with the next time slot
    for i in range(len(newlist)-4): #ignore the last one
        if newlist1[i] == []:
            newindex = 0
            while newlist1[i+newindex] == []:
                newindex = newindex +1
            newlist1[i] = newlist1[i+newindex]

    # merge lefthand and right hand together
    finallist = []
    for i in range(len(newlist)): #two lists should be in the same length
        finallist.append(''.join(newlist[i]))
        finallist.append(''.join(newlist1[i]))
        finallist.append(" ")
        
    finallist = ''.join(finallist) #collapse the final list
    return finallist #it is actually a string
        
    
       
              
#sort one hand #NOT USED
def sorttime(righthand, lefthand, timescale):  
##    print(righthand[1])
##    print(lefthand[1])
    newlist = []
    #if the times are equal merge
    i = 0
    flag1 =False
    flag2 = False
    flag3 = False
    temp_time = 0;
    for i in range(len(righthand[0])-3): 
        if (righthand[1][i] == righthand[1][i+1]) and (righthand[1][i] == righthand[1][i+2]) and (righthand[1][i] == righthand[1][i+3]):
            newlist.append(str(chr(righthand[0][i])))
            newlist.append(str(chr(righthand[0][i+1])))
            newlist.append(str(chr(righthand[0][i+2])))
            newlist.append(str(chr(righthand[0][i+3])))
            newlist.append(" ")
            flag1 = True
        elif (righthand[1][i] == righthand[1][i+1]) and (righthand[1][i] == righthand[1][i+2]):
            newlist.append(str(chr(righthand[0][i])))
            newlist.append(str(chr(righthand[0][i+1])))
            newlist.append(str(chr(righthand[0][i+2])))
            newlist.append(" ")
            flag2 = True
        elif (righthand[1][i] == righthand[1][i+1]):
            newlist.append(str(chr(righthand[0][i])))
            newlist.append(str(chr(righthand[0][i+1])))
            newlist.append(" ")
            flag3 = True
        else:
            newlist.append(str(chr(righthand[0][i])))
            newlist.append(" ")

##    if flag1 == False and flag2 == False and flag3 == False: #no stop
##    if flag1 == False and flag2 == False and flag3 == True: #double stop
##    if flag1 == False and flag2 == True and flag3 == True:#triple stop
##    if flag1 == true and flag2 == True and flag3 == True:#fourth stop
    if(righthand[0][i+1] == righthand[0][i+2]) & (righthand[0][i+1] == righthand[0][i+3]) :        
        newlist.append(str(chr(righthand[0][i+1])))
        newlist.append(str(chr(righthand[0][i+2])))
        newlist.append(str(chr(righthand[0][i+3])))
    if(righthand[0][i+1] == righthand[0][i+2]) :       
        newlist.append(str(chr(righthand[0][i+1])))
        newlist.append(str(chr(righthand[0][i+2])))
        newlist.append(" ")
        newlist.append(str(chr(righthand[0][i+3])))
    else:
        newlist.append(str(chr(righthand[0][i+1])))
        newlist.append(" ")
        newlist.append(str(chr(righthand[0][i+2])))
        newlist.append(" ")
        newlist.append(str(chr(righthand[0][i+3])))       
       
##    print(''.join(newlist) )
##    print(len(newlist))
    print(newlist[0:100])
    return newlist



def main():
    readinfiles()


''' take a 2-d list and return the flattened 1-d list'''
def flatten(l):
    try:
        return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    except IndexError:
        return []

'''clear the idle screen '''
def cls():
    print ("\n" * 100)

    
    
main()


