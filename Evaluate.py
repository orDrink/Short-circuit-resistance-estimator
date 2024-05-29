from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from numpy import *
import math

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

root_url = "/users/yjia3/ml_risk_cycled/"
root_url_data = root_url+"DATA/"
root_url_model = root_url+"MODEL/V6/"
test_file_name="R500N300"
model_name_0="RNN_"+test_file_name
file_name=test_file_name

channel = 7
features = [[0,1,2,3,4,5,6]]
# features = [[0,1,2,3,4],[0,1,2,3],[0,1,2,3,5],[0,1,2,4],[0,1,2,4,6]]

test_size = 0.1
sigement_sizes=[120]
# sigement_sizes=[60,120,180,300,600]

smallest_size = 60
repeated_time = 5

for sigement_size in sigement_sizes:

  print(sigement_size)
  input0 = np.empty(shape=(0,channel,sigement_size))
  label = np.empty(shape=(0))

  samples = np.loadtxt(open(root_url_data+file_name+"T"+str(sigement_size)+"_S.csv","rb"),delimiter=",",skiprows=0)
  input0 = np.reshape(samples[:,0:sigement_size*channel],(len(samples),channel,sigement_size))
  label = samples[:,sigement_size*channel]
  input0 = input0.transpose(0,2,1)

  reference_resistance = 100
  y0 = np.log(reference_resistance/1e6)
  y1 = np.log(reference_resistance/1e-1)
  label = (label-y0)/(y1-y0)

  for feature in features:

    print(feature)

    input = input0[:,:,feature]
    print(np.shape(input))

    for repeat in range(repeated_time):

      print(str(repeat)+"/"+str(repeated_time)+" testing")

      ouput_file_name=file_name+'T'+str(sigement_sizes[-1])+'T'+str(repeat)
      testfile = open(root_url_model+ouput_file_name+".csv","w", encoding='utf-8', newline='')
      writer = csv.writer(testfile)

      idx = np.random.RandomState(seed=repeat).permutation(len(label))
      input_ramdom = input[idx]
      label_ramdom = label[idx]

      cut_idx = int(round((1-test_size) * len(label)))
      x_train, x_test = input_ramdom[:cut_idx], input_ramdom[cut_idx:]
      y_train, y_test = label_ramdom[:cut_idx], label_ramdom[cut_idx:]

      if sigement_size < smallest_size:
        x_test = layers.UpSampling1D(size=round(smallest_size/sigement_size))(x_test)   

      # input_shape=x_train.shape[1:]
      # print(np.shape(x_train))

      model_name = model_name_0+"T"+str(sigement_size)+str(repeat)+"_"
      for feature0 in feature:
        model_name = model_name+str(feature0)

      model = keras.models.load_model(root_url_model+model_name)
      # model.summary()

      y_predictions = model.predict(x_test).flatten()
      
      number_test = len(y_predictions)
      l1 = 0
      l2 = 0
      for i in range(number_test):

        current=x_test[i,:,2].flatten()
        #print(current)
        std_i=np.std(current)
        avg_i=np.mean(current)

        C_current = 0
        Group = 1

        if std_i/avg_i<0.01:
          Group = 0
        C_current = round(avg_i/3.2*10)/10

        r_prediction = reference_resistance*math.exp(-y_predictions[i]*(y1-y0)-y0)
        r_test = reference_resistance*math.exp(-y_test[i]*(y1-y0)-y0)

        writer.writerow([Group, std_i, avg_i,  C_current, y_test[i], y_predictions[i], r_prediction, r_test])
      
      testfile.close()     
