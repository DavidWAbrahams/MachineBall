from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from training_helpers import LoadData, ShuffleCallback

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()

parser.add_argument('--sample_path', action='store',
                    default='.\\samples.p', dest='sample_path',
                    help='Input path for training sample pickle')
parser.add_argument('--label_path', action='store',
                    default='.\\labels.p', dest='label_path',
                    help='Input path for training label pickle')
parser.add_argument('--model_path', action='store',
                    default='winner_model.h5', dest='model_path',
                    help='Output path for the saved model')
parser.add_argument('--max_epochs', action='store',
                    default=1000, dest='max_epochs',
                    help='Max training epochs', type=int)
parser.add_argument('--batch_size', action='store',
                    default=1000, dest='batch_size',
                    help='Max training epochs', type=int)
                    
args = parser.parse_args()

# Hack to bypass CuBlas errors
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

samples_train = None
samples_test = None
labels_train = None
labels_test = None

# Read and pad data from disk
if os.path.isfile(args.sample_path) and os.path.isfile(args.label_path):
  print('Using labeled data found at {}'.format(args.sample_path))
  samples_train, samples_test, labels_train, labels_test = LoadData(
    args.sample_path, args.label_path, test_fraction=0.1)
  # Label data comes in the form [visitors score, home score].
  # Condense to just a winner (0=visitors, 1=home).
  labels_train = np.asarray(
    [points[0] > points[1] for points in labels_train], dtype=bool)
  labels_test = np.asarray(
    [points[0] > points[1] for points in labels_test], dtype=bool)
else:
  raise Exception('Unable to find processed data. Please run parser.py first, or use --sample_path and --label_path if data are not in the default location.')

# Early stopping with patience
early_stopper = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
model_checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss',
  mode='min', save_best_only=True, verbose=1)
# Shuffle player order in each epoch
shuffler = ShuffleCallback(samples_train)
  
# Define and train the model
input_shape = samples_train[0].shape
model = Sequential()
model.add(Bidirectional(LSTM(
  128, return_sequences=True, input_shape=input_shape,
  dropout=0.4, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(
  128, return_sequences=True)))
model.add(Bidirectional(LSTM(
  128)))
model.add(Dropout(0.4))
model.add(Dense(
  1, activation='sigmoid',
  kernel_regularizer=regularizers.l2(0.04),
  bias_regularizer=regularizers.l2(0.04)))
model.compile(
  loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(samples_train, labels_train,
  epochs=args.max_epochs, batch_size=args.batch_size,
  validation_split=0.1, shuffle=True,
  callbacks=[early_stopper, model_checkpoint, shuffler])

print('')
print('*********************************')
print('Testing best model on unseen data')
model = load_model(args.model_path)
_, accuracy = model.evaluate(x=samples_test, y=labels_test)
print('Test accuracy: {:.2f}'.format(accuracy*100))