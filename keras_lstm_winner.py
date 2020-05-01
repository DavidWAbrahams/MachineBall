import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from keras.callbacks.callbacks import Callback, EarlyStopping, ModelCheckpoint

import argparse
import os
import pickle

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
                    
args = parser.parse_args()

samples_train = None
labels_train = None

samples_test = None
labels_test = None

# Hack to bypass CuBlas errors
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def ShufflePlayers(samples):
  # Shuffle the order of players within each game. This prevents the model from
  # unfairly knowing who will start the game vs who will be substituted later.
  rng = np.random.default_rng()
  rng.shuffle(samples, axis=1)
  
class ShuffleCallback(Callback): 
  """Shuffling the order of players just once is a disaster for the
  model's reported performance. The model is denied unfair knowledge about who
  is starting so it performs way worse. But if we shuffle at each epoch it
  acts like data augmentation, allowing us to train MUCH longer before 
  overfitting."""
  def on_epoch_begin(self, epoch, logs={}):
    ShufflePlayers(samples_train)

# Early stopping callbacks
early_stopper = EarlyStopping(monitor='val_loss', verbose=1, patience=50)
model_checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# Massage the data into numpy arrays
if os.path.isfile(args.sample_path) and os.path.isfile(args.label_path):
  print('Using labeled data found at {}'.format(args.sample_path))
  samples = pickle.load(open(args.sample_path, 'rb'))
  print('Read {} games'.format(len(samples)))
  max_num_players = max([len(s) for s in samples])
  
  # Pad all game samples to the same length
  player_len = len(samples[-1][0])
  for game in samples:
    while len(game) < max_num_players * 1.1:
      game.append([0]*player_len)
  samples = np.asarray(samples)
  ShufflePlayers(samples)
  
  labels = np.array(pickle.load(open(args.label_path, 'rb')))
  labels = np.asarray([l[0] > l[1] for l in labels], dtype=bool)
  
  # Hold back some data for testing
  labels_train, labels_test = np.split(labels, [int(.9 * len(labels))])
  samples_train, samples_test = np.split(samples, [int(.9 * len(samples))])
else:
  raise Exception('Unable to find processed data. Please run parser.py first, or use --sample_path and --label_path if data are not in the default location.')

# Define and train the model
input_shape = samples_train[0].shape
model = Sequential()
model.add(Bidirectional(LSTM(
  128, return_sequences=True, input_shape=input_shape,
  dropout=0.1, recurrent_dropout=0.1)))
model.add(Bidirectional(LSTM(
  128, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)))
model.add(Bidirectional(LSTM(
  128, dropout=0.0, recurrent_dropout=0.0)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(samples_train, labels_train, epochs=1000, batch_size=1000, validation_split=0.1, shuffle=True, callbacks=[early_stopper, model_checkpoint, ShuffleCallback()])

print('')
print('*********************************')
print('Testing best model on unseen data')
model = load_model(args.model_path)
_, accuracy = model.evaluate(x=samples_test, y=labels_test)
print('Test accuracy: %.2f' % (accuracy*100))