'''
CSC420 - Project 2
Zeyang Huang
Lab Partner: Rao Fu
'''
'''
select 100 pages randomly
manually classify: recipie or non-recipe
extract just the text from 100
strip out all non-alphabetical characters
collapse whitespace (that is, replace multiple spaces, tabs, newlines, etc. with just a single space character) 

Convert the strings with the extracted words into a sparse matrix
Split the data into a training and evaluation set
Apply Multinomial Naïve Bayes
Generate your predicted class labels for the test data and compare these to ground truth
Generate evaluation metrics, including a confusion matrix and F1 score 
'''
import random
import re
import sys
import os
import numpy as np
#import the build-in dataset loader for 20 newsgroups
from sklearn.datasets import fetch_20newsgroups
#tokenizing and filtering of stopwords
from sklearn.feature_extraction.text import CountVectorizer
#term frequencies
from sklearn.feature_extraction.text import TfidfTransformer
#naïve Bayes classifier suitable for word counts - multinomial variant
from sklearn.naive_bayes import MultinomialNB
#a pipeline class like a compound classifier
from sklearn.pipeline import Pipeline
#a linear support vector machine (SVM)
from sklearn.linear_model import SGDClassifier
#detailed performance analysis results
from sklearn import metrics
#parameter tuning using grid search
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

''' this function reads the input file line by line, detects if the <page> tag is found returns the contents between the <page> and </page> tag & the title '''
def getNextPage(inputfile):
    line = inputfile.readline()
    page = line  
    while line != "" and "<page>" not in line: #keep read lines until <page> is found
        line = inputfile.readline()   
    line = inputfile.readline()     #assumes <title> always fallows the <page> tag
    title = line
    page = page + title   
    while line != "" and "</page>" not in line: #stops adding lines to the page when </page> is found
        line = inputfile.readline()
        page = page + line      
    return page, title

''' this function detects if the title contains a colon returns the page if its title does not have a colon'''
def getNextContentPage(inputfile):
    try:
        page,title = getNextPage(inputfile)
        match = re.findall("<title>(.*?)</title>", title)[0]   
        while ":" in match and title != '':  #keep running until the page's title does not contain a colon
            page,title = getNextPage(inputfile)
            match = re.findall("<title>(.*?)</title>", title)[0]
    except IndexError: 
        match = ''
    return page,title
      
''' uses Knuth's algorithm and output the selected pages to "data.xml" '''
def randomPageExtract(inputfile):
    track = []
    outputfile = open("data.xml", "w")
    m = 0  #number of pages haven't selected
    n = 50  #number of selection
    N = 144534 - 94083 #number of content pages    
    for t in range (N):       
        page,title = getNextContentPage(inputfile)
        u = random.random()
        if u < ((n-m)/(N-t)):
            track.append(t)
            outputfile.write(page)
            m = m+1          
    outputfile.close()

''' Count the number of content pages return the int count '''
def contentPageCount(inputfile):
    count = 0
    page, title = getNextContentPage(inputfile)
    while page != '':
        count = count +1
        page, title = getNextContentPage(inputfile)      
    return count

'''extract the text from content page's tags, return the raw text and previously manully classified labels '''
def extractPageText(inputfile):
    category = [] #the array for the classified labels
    line = inputfile.readline()#assume the first line is always <page> 
    line = inputfile.readline() #assume the next line is always <title>
    match = re.findall("<.*?>(.*?)</.*?>", line)[0] #extract the title content from the tag
    page = match #store the text in the same variable
    
    line = inputfile.readline() #assume the next line is always the manully classified category labels
    match = re.findall("<.*?>(.*?)</.*?>", line)[0] #extract the content from tag
    match_category = match[9:] #assume the content all follows the format "labeled: \d*" 
    category.append(match_category) #append the labels in a array
   
    while line != "" and "</page>" not in line: #stops adding lines to the page when </page> is found
        line = inputfile.readline()
        #if there is a format <tag> content </tag>, then extract the content
        match = re.findall("<\w.*?>(.*?)</.*?>", line)
        match_start = re.findall("<\w.*?>", line)
        if match: 
            page = page+" "+match[0]
        #if the content has multiple lines:
        elif match_start:
            line = inputfile.readline()
            match = re.findall("<\w.*?>(.*?)</.*?>", line) #extract the first line
            if match:
                page = page+" "+match[0] #add to the page
            #while does not find the end tag </*d>              
            match_end = re.findall("</.*?>", line)
            while not match_end:
                line = inputfile.readline() 
                line = line.rstrip()
                #the senario that content and end tag in the same line
                match_text_w_end = re.findall("(.*?)</.*?>", line) 
                if match_text_w_end: #if  content </endtag> #just extract the content
                    page = page + match_text_w_end[0]
                else: #else extract the whole line
                    page = page + line                  
                match_end = re.findall("</.*?>", line)
    return category, page
            

''' extract the content text for every content page in the input file return a list of text and a list of category ''' 
def getalltext(inputfile,size):
    categorylist = []
    pagelist = []
    for i in range(size): #assume size is consistant with the number of pages in the input file
        category, page = extractPageText(inputfile)
        categorylist.append(category[0])
        pagelist.append(page)      
    return categorylist,pagelist


''' replace all non-alphabetical character to whitespace and collapse all whitespace '''
def refinecharacterintext(inputfile,size):
    categorylist,pagelist = getalltext(inputfile,size)
    newpagelist = []
    for index in range(len(categorylist)):
        newString = ""
        for i in range(len(pagelist[index])):
            if not pagelist[index][i].isalpha(): #if not isalpha() add a whitespace 
                    newString =  newString + " "
            else:
                newString =  newString + pagelist[index][i]
        newString = ' '.join(newString.split()) #collapse several white space to only one
        newpagelist.append(newString)      
    return categorylist,newpagelist

''' FIVE CLASSES: dessert, drink, main, vege, n '''   
''' training the classifier using multinominal naive bayes and k-fold cross validation '''
''' import the test xml and test the final classifier'''
def training(inputfile):
##    size = contentPageCount(inputfile) #synchronize? #python somehow cannot run function one by one
    categories = ["dessert","drink","main","vege","n"] #the five predicted categories
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()  #import transformer to turn the string into vectors

    categorylist,pagelist = refinecharacterintext(inputfile,100) #get the data 
    newcategorylist = transformcategory(categorylist, categories) #transform the classification with the according index number eg: dessert = 0, drink = 1 
    categorylist = newcategorylist

    #k-fold using the package from tutorial found on line: https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/
    test_size = 0.33
    seed = 7
    kfold = model_selection.ShuffleSplit(n_splits=10, test_size=test_size, random_state=seed)
    X_train_counts = count_vect.fit_transform(pagelist)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, categorylist) 
    results = model_selection.cross_val_score(clf, X_train_tfidf, categorylist, cv=kfold)
    ##    print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
   
    #k-fold cross-validation:
    k_fold = KFold(n_splits=10, random_state=None, shuffle=True) #n-splits is the k for k-fold cross validation #shuffle = true: make it random
    pred_score = 0 #the prediction accuracy score for the trained classifier on the validation set

    #split the 100 selected data randomly into 60% training and 40% validation sets 10 times 
    for train_indices, test_indices in k_fold.split(pagelist, categorylist):
        newpagelist = [] #new training set for x
        newcategorylist = [] #new training set for y
        newpagelist_test = [] #new validation set for x
        newcategorylist_test = [] #new validation set for y     
        #iterate the train_indices list with randomly selected indexes and map them with the actual strings in the list to create a new sublist
        for i in range(len(train_indices)):  #training set
            newpagelist.append(pagelist[train_indices[i]])
            newcategorylist.append(categorylist[train_indices[i]])
            X_train_counts = count_vect.fit_transform(newpagelist)
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        for i in range(len(test_indices)): #testing set (validation)
            newpagelist_test.append(pagelist[test_indices[i]])
            newcategorylist_test.append(categorylist[test_indices[i]])
            X_counts_test = count_vect.transform(newpagelist_test)
            X_tfidf_test = tfidf_transformer.transform(X_counts_test)
        #train the classifier and do the prediction on the validation set
        clf = MultinomialNB().fit(X_train_tfidf, newcategorylist)          
        #predicted = clf.predict(X_tfidf_test)
        #print(np.mean(predicted == newcategorylist_test))

                   
    #simple test #just an example to show the prediction class for a example input
    docs_example = ["soymilk milk sugar flour bake","noodle stew potato tomato","this is a category"]
    X_example_counts = count_vect.transform(docs_example)
    X_example_tfidf = tfidf_transformer.transform(X_example_counts)
    predicted_example = clf.predict(X_example_tfidf)
    print(predicted_example)
    '''[0 2 4]'''
    '''['dessert' 'main' 'n']'''
    
    #testing with the brandnew 50 labeled dataset
    input50test = open("data_test.xml","r")
    categorylist_test, pagelist_test = refinecharacterintext(input50test,50)
    newcategorylist_test = transformcategory(categorylist_test, categories)
    categorylist_test = newcategorylist_test 
    #transform the data 
    X_test_counts = count_vect.transform(pagelist_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    predicted = clf.predict(X_test_tfidf)
    #print the result
    print("Testing set Accuracy: ",np.mean(predicted == categorylist_test))
    print(metrics.classification_report(categorylist_test, predicted, target_names=categories))
    print(metrics.confusion_matrix(categorylist_test, predicted))
    input50test.close() #close the file


'''THREE CLASSES: SIDE, MAIN, NON-RECIPE '''
'''train, test '''
def traininglessclass(inputfile):
    categories = ["dessert","drink","main","vege","n"] #the five predicted categories
    newcategories = ["side","main","non-recipe"]
    categorylist,pagelist = refinecharacterintext(inputfile,100)
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()  #import transformer to turn the string into vectors
    newcategorylist = transformcategory(categorylist, categories) #transform the classification with the according index number eg: dessert = 0, drink = 1 
    categorylist = combineCategories(newcategorylist) #change the previous 5 classes to 3 classes
    X_train_counts = count_vect.fit_transform(pagelist)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    seed = 7 #random state
    kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
    categorylist = np.array(categorylist) 
    clf = MultinomialNB().fit(X_train_tfidf, categorylist) #train the data
    print("the accuracy over the training set: ",clf.score(X_train_tfidf,categorylist))
   
    #testing with the brandnew 50 labeled dataset
    input50test = open("data_test.xml","r")
    categorylist_test, pagelist_test = refinecharacterintext(input50test,50) #get the 5 classes data
    newcategorylist_test = transformcategory(categorylist_test, categories) #transform the string label to index
    categorylist_test = combineCategories(newcategorylist_test) #transform 5 class labels to 3 class labels
    categorylist_test = np.array(categorylist_test) #into np array
    #transform the data 
    X_test_counts = count_vect.transform(pagelist_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts) #transform into matrix
    predicted = clf.predict(X_test_tfidf) #do the prediction by the trained classifier

    #print the result
    print("the accuracy over the testing set: ",np.mean(predicted == categorylist_test))
    print(metrics.classification_report(categorylist_test, predicted, target_names=newcategories))
    print(metrics.confusion_matrix(categorylist_test, predicted))
    input50test.close() #close the file


    

'''just have three classes dessert, main, non-recipe '''
def combineCategories(categorylist):
    newcategorylist = []
    for i in range(len(categorylist)):
        if  categorylist[i] == 2 or categorylist[i] == 3:
            newcategorylist.append(1)
        if categorylist[i] == 4:
            newcategorylist.append(2)
        if categorylist[i] == 0 or categorylist[i] == 1 :
            newcategorylist.append(0)
    return newcategorylist
            

'''iterate the list and change the string class labels to the index'''
def transformcategory(categorylist, categories):
    newcategorylist = []
    for i in range(len(categorylist)):
        for t in range(len(categories)):
            if categorylist[i] == categories[t]:
                newcategorylist.append(t)               
    return newcategorylist


def main():
##    inputfile = open("cookbook.xml","r")
##    randomPageExtract(inputfile) #extract 100 random pages from the dataset
##    inputfile.close()

    input100random = open("data_100.xml","r")
    #cannot uncomment both functions below and run them at same time...
    #may need the keyword like "synchronized" in java
    
##    training(input100random) #five class training 
    traininglessclass(input100random) #3 class training
    input100random.close()
    print("\n"+"---running done----")


def cls():
    print ("\n" * 100)
    

main()
