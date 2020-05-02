from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Bidirectional, Dropout
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from training_helpers import ShufflePlayers, LoadData, ShuffleCallback
import numpy as np

import argparse
import os

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
model.add(Bidirectional(GRU(64, return_sequences=True, input_shape=input_shape, dropout=0.1, recurrent_dropout=0.1)))
model.add(Bidirectional(GRU(64, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)))
model.add(Bidirectional(GRU(64, dropout=0.0, recurrent_dropout=0.0)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(samples_train, labels_train,
  epochs=args.max_epochs, batch_size=args.batch_size,
  validation_split=0.07, shuffle=True,
  callbacks=[early_stopper, model_checkpoint, shuffler])

print('')
print('*********************************')
print('Testing best model on unseen data')
model = load_model(args.model_path)
predicted_scores = np.squeeze(model.predict(x=samples_test))
predicted_winners = np.array([score[1]>score[0] for score in predicted_scores], dtype=int)
actual_winners = np.array([score[1]>score[0] for score in labels_test], dtype=int)
wrong_winners = np.sum(np.abs(np.subtract(predicted_winners, actual_winners)))
print('Accuracy predicting game winners on test data: {:.2f}'.format(
  (len(actual_winners)-wrong_winners)/len(actual_winners)*100))
wrong_by_avg = 2 * np.mean(np.abs(np.subtract(predicted_scores, labels_test)))
print(
  'The model\'s score was wrong by an average of {:.2f} points.'.format(
  wrong_by_avg))
wrong_by_med = 2 * np.median(np.abs(np.subtract(predicted_scores, labels_test)))
print('The model\'s score was wrong by a median of {:.2f} points.'.format(
  wrong_by_med))
print('Example games\' actual score vs predicted:')
for i in range(10):
  print('{}\tvs\t{}'.format(labels_test[i], predicted_scores[i]))
