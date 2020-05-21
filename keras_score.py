from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from training_helpers import LoadData, ShuffleCallback, TrainingArgs

import numpy as np
import os
                    
args = TrainingArgs()

# Hack to bypass CuBlas errors
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

x_train = None
x_test = None
y_train = None
y_test = None

# Read and pad data from disk
print('Using labeled data found at {}'.format(args.parsed_data_prefix))
x_train, x_validate, x_test, y_train, y_validate, y_test = LoadData(
  args.parsed_data_prefix,
  drop_fraction=args.drop_fraction, test_drop_fraction=args.test_drop_fraction,
  validate_fraction=args.validate_fraction, test_fraction=args.test_fraction)
  
# Early stopping with patience
early_stopper = EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience)
model_checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss',
  mode='min', save_best_only=True, verbose=1)
callbacks = [early_stopper, model_checkpoint]
if args.roster_shuffle:
  print('Roster shuffling (data augmentation) enabled.')
  callbacks.append(ShuffleCallback(x_train))
  
# Define and train the model
input_shape = x_train[0].shape
model = Sequential()
model.add(Bidirectional(GRU(args.rnn_layer_size, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2)))
for _ in range(args.num_rnn_layers - 2):
  model.add(Bidirectional(GRU(args.rnn_layer_size, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)))
model.add(Bidirectional(GRU(args.rnn_layer_size, dropout=0.0, recurrent_dropout=0.0)))
model.add(Dropout(0.4))
model.add(Dense(
  2, activation='linear',
  kernel_regularizer=regularizers.l2(0.04),
  bias_regularizer=regularizers.l2(0.04)))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
  epochs=args.max_epochs, batch_size=args.batch_size,
  validation_data=(x_validate, y_validate), shuffle=True,
  callbacks=callbacks)

print('')
print('*********************************')
print('Testing best model on unseen data')
model = load_model(args.model_path)
predicted_scores = np.squeeze(model.predict(x=x_test))
predicted_winners = np.array([score[1]>score[0] for score in predicted_scores], dtype=int)
actual_winners = np.array([score[1]>score[0] for score in y_test], dtype=int)
wrong_winners = np.sum(np.abs(np.subtract(predicted_winners, actual_winners)))
print('Accuracy predicting game winners on test data: {:.2f}'.format(
  (len(actual_winners)-wrong_winners)/len(actual_winners)*100))
wrong_by_avg = 2 * np.mean(np.abs(np.subtract(predicted_scores, y_test)))
print(
  'The model\'s score was wrong by an average of {:.2f} points.'.format(
  wrong_by_avg))
wrong_by_med = 2 * np.median(np.abs(np.subtract(predicted_scores, y_test)))
print('The model\'s score was wrong by a median of {:.2f} points.'.format(
  wrong_by_med))
print('Example games\' actual score vs predicted:')
for i in range(10):
  print('{}\tvs\t{}'.format(y_test[i], predicted_scores[i]))
