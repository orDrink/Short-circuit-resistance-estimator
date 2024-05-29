from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from numpy import *

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

root_url = "/users/yjia3/ml_risk_cycled/"
root_url_data = root_url+"DATA/"
root_url_model = root_url+"MODEL/"

channel = 7
sigement_size=180

file_names= [
   "R000N001","R000N001_2","R000N050","R000N050_2","R000N100",
   "R000N100_2","R000N200","R000N200_2","R000N300","R000N300_2",
   "R050N050","R050N100","R050N200","R050N300","R050N001",
   "R100N300","R100N200","R100N100","R100N050","R100N001",
   "R200N300","R200N200","R200N100","R200N050","R200N001",
   "R500N300","R500N200","R500N100","R500N050","R500N001"
   ]

sample = np.empty(shape=(0,channel*sigement_size+1))

for file_name in file_names:
  samples0 = np.loadtxt(open(root_url_data+file_name+"_S.csv","rb"),delimiter=",",skiprows=0)
  print(np.shape(samples0))
  sample = np.vstack((sample,samples0)) 

print(np.shape(sample))

idx = np.random.permutation(len(sample))
sample_ramdom = sample[idx]

test_size = 0.1  #[CHANGE]
cut_idx = int(round((1-test_size) * len(sample)))
sample_train, sample_test = sample_ramdom[:cut_idx], sample_ramdom[cut_idx:]

file_output="TRAINING"
samplefile = open(root_url_data+file_output+".csv","w", encoding='utf-8', newline='')
writer = csv.writer(samplefile)
data_len=len(sample_train)
for i in range(data_len):
  writer.writerow(sample_train[i])
samplefile.close()

file_output="TESTING"
samplefile = open(root_url_data+file_output+".csv","w", encoding='utf-8', newline='')
writer = csv.writer(samplefile)
data_len=len(sample_test)
for i in range(data_len):
  writer.writerow(sample_test[i])
samplefile.close()
