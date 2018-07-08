Due May 12, 2018
Files Uploaded: csvprocess.py  lstmtrain.py  makeprediction.py postmidiprocessing.py  postmidiprocessing_v2.py
1. Run "csvprocess.py" which takes 40 csv file named from "EP1.csv" to "EP40.csv" and output the ascii transformations of the data into "output.txt"
2. Run "lstmtrain.py" which takes an txt file named "output.txt" as the input to the neural net. It will train the LSTM neural net and locally save the model as "model.yaml" and save the best weights for each epoch as "weights-epoch number-loss.hdf5"
3. Run "makepredictions.py" that takes "output.txt", model and weights' file path to reload the model and make predictions
4. Run "postmidiprocessing.py" that uses the prediction string and transform into csv so that you could manually combine the csv and transform it back to midi using http://www.fourmilab.ch/webtools/midicsv/ or you could run "postmidiprocessing_v2.py" that take a different pattern of a string and transform it to csv.

