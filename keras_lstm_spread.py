import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint

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
                    default='spread_model.h5', dest='model_path',
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

# Early stopping callbacks
early_stopper = EarlyStopping(monitor='val_loss', verbose=1, patience=30)
model_checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# Massage the data into numpy arrays
if os.path.isfile(args.sample_path) and os.path.isfile(args.label_path):
  print('Using labeled data found at {}'.format(args.sample_path))
  samples = pickle.load(open(args.sample_path, 'rb'))
  print('Read {} games'.format(len(samples)))
  max_len = max([len(s) for s in samples])
  min_len = min([len(s) for s in samples])
  print('Game rosters vary in length from {} to {} players'.format(min_len, max_len))
  player_len = len(samples[len(samples)//2][0])
  print('Each player has {} stats'.format(player_len))
  
  # Pad all game samples to the same length
  for game in samples:
    while len(game) < max_len * 1.1:
      game.append([0]*player_len)
  samples = np.asarray(samples)
  
  labels = np.array(pickle.load(open(args.label_path, 'rb')))
  labels = np.asarray([l[0] - l[1] for l in labels], dtype=int)
  
  # Hold back some data for testing
  labels_train, labels_test = np.split(labels, [int(.9 * len(labels))])
  print('Training labels shape: {}'.format(labels_train.shape))
  samples_train, samples_test = np.split(samples, [int(.9 * len(samples))])
  print('Training sample shape: {}'.format(samples_train.shape))
else:
  raise Exception('Unable to find processed data. Please run the parser first, or use --sample_path and --label_path if they are not in the default location.')

# Define and train the model
input_shape = samples_train[0].shape
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape, dropout=0.1, recurrent_dropout=0.1)))
model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)))
model.add(Bidirectional(LSTM(64, dropout=0.0, recurrent_dropout=0.0)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(samples_train, labels_train, epochs=250, batch_size=1000, validation_split=0.1, shuffle=True, callbacks=[early_stopper, model_checkpoint])

print('')
print('*********************************')
print('Testing best model on unseen data')
model = load_model(args.model_path)
predicted_spreads = np.squeeze(model.predict(x=samples_test))
predicted_winners = np.array([spread < 0 for spread in predicted_spreads], dtype=int)
actual_winners = np.array([spread < 0 for spread in labels_test], dtype=int)
wrong = np.sum(np.abs(np.subtract(predicted_winners, actual_winners)))
print('Accuracy predicting game winners on test data: %.2f' % ((len(actual_winners)-wrong)/len(actual_winners)*100))

wrong_by = np.mean(np.abs(np.subtract(predicted_spreads, labels_test)))
print('The model\'s spread was wrong by an average of {:.2f} points.'.format(wrong_by))
wrong_by = np.median(np.abs(np.subtract(predicted_spreads, labels_test)))
print('The model\'s spread was wrong by a median of {:.2f} points.'.format(wrong_by))
