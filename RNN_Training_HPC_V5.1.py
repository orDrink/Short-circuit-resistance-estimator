from tensorflow import keras
#import torch
#from torch import nn
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
test_file_name="R500N300"
model_name_0="RNN_"+test_file_name
file_name=test_file_name

channel = 7
features = [[0,1,2,3,4,5,6]]
# features = [[0,1,2],[0,1,2,3,4],[0,1,2,3],[0,1,2,3,5],[0,1,2,4],[0,1,2,4,6]]
# sigement_size=180
test_size = 0.1  #[CHANGE]
reference_resistance = 100
epochs = 200
batch_size = 128
repeated_time  =  5
smallest_size = 60

# sigement_sizes=[60,120,180,300,600]
sigement_sizes=[120]
# sigement_sizes=[10, 20, 30, 60, 120, 300]

for sigement_size in sigement_sizes:

  print(sigement_size)

  input0 = np.empty(shape=(0,channel,sigement_size))
  label = np.empty(shape=(0))

  samples = np.loadtxt(open(root_url_data+file_name+"T"+str(sigement_size)+"_S.csv","rb"),delimiter=",",skiprows=0)
  input0 = np.reshape(samples[:,0:sigement_size*channel],(len(samples),channel,sigement_size))
  label = samples[:,sigement_size*channel]
  input0 = input0.transpose(0,2,1)

  for feature in features:

    print(feature)

    input = input0[:,:,feature]
    print(np.shape(input))

    y0 = np.log(reference_resistance/1e6)
    y1 = np.log(reference_resistance/1e-1)
    label = (label-y0)/(y1-y0)

    for repeat in range(repeated_time):

      print("")
      print("======================================================")
      print(str(repeat)+"/"+str(repeated_time)+" training")

      idx = np.random.RandomState(seed=repeat).permutation(len(label))
      input_ramdom = input[idx]
      label_ramdom = label[idx]

      cut_idx = int(round((1-test_size) * len(label)))
      x_train, x_test = input_ramdom[:cut_idx], input_ramdom[cut_idx:]
      y_train, y_test = label_ramdom[:cut_idx], label_ramdom[cut_idx:]

      norm_layer = layers.experimental.preprocessing.Normalization(axis=-1)
      adapt_data=input
      norm_layer.adapt(adapt_data)

      if sigement_size < smallest_size:
  #      x_train = nn.functional.upsample(x_train, size=(x_train.shape[0],60,x_train.shape[2]), mode='linear', align_corners=True)
        x_train = layers.UpSampling1D(size=round(smallest_size/sigement_size))(x_train)    
      input_shape=x_train.shape[1:]
      print(np.shape(x_train))

      model = models.Sequential([
          layers.Input(shape=input_shape),
          norm_layer,

#          layers.Conv1D(64, 2, activation='relu'),
          
#          layers.Conv1D(64, 3, activation='relu'),
#          layers.Conv1D(64, 3, activation='relu'),
#          layers.Conv1D(64, 3, activation='relu'),
#          layers.Conv1D(64, 3, activation='relu'),
          
#          layers.Dropout(0.25),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),

#          layers.Dropout(0.5),

          layers.Dense(1),
      ])

      model.summary()

      callbacks = [
          keras.callbacks.ModelCheckpoint(
              "best_model.h5", save_best_only=True, monitor="val_loss"
          ),
          keras.callbacks.ReduceLROnPlateau(
              monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
          ),
          # keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
      ]

      optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
      
      model.compile(
          optimizer=optimizer,
          loss="mse",
          metrics=["mae","mse"],
      )

      history = model.fit(
          x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_split=0.2,
          verbose=2
      )

      model_name = model_name_0+"T"+str(sigement_size)+str(repeat)+str(feature)
      model.save(root_url_model+model_name)

      metric = "mse"
      historyfile = open(root_url_model+model_name+".csv","w", encoding='utf-8', newline='')
      writer = csv.writer(historyfile)

      for step in range(len(history.history[metric])):
        writer.writerow([step,history.history[metric][step],history.history["val_" + metric][step]])
      historyfile.close()

      plt.figure()
      plt.plot(history.history[metric])
      plt.plot(history.history["val_" + metric])
      plt.title("model " + metric)
      plt.ylabel(metric, fontsize="large")
      plt.xlabel("epoch", fontsize="large")
      plt.legend(["train", "val"], loc="best")
      plt.savefig(root_url_model+model_name+".png")
      # plt.show()
      plt.close()