from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization, Flatten, Conv1D, Activation
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from training_helpers import LoadData, ShuffleCallback, TrainingArgs

import numpy as np
import os
import matplotlib.pyplot as plt
                    
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
# Label data comes in the form [visitors score, home score].
# Condense to just a winner (0=visitors, 1=home).
y_train = np.asarray(
  [points[1] > points[0] for points in y_train], dtype=bool)
y_validate = np.asarray(
  [points[1] > points[0] for points in y_validate], dtype=bool)
y_test = np.asarray(
  [points[1] > points[0] for points in y_test], dtype=bool)
  
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
for _ in range(args.num_rnn_layers - 1):
  model.add(Bidirectional(LSTM(
    args.rnn_layer_size, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)))
model.add(Bidirectional(LSTM(args.rnn_layer_size)))
model.add(Dropout(0.4))
model.add(Dense(args.rnn_layer_size//2, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.3))

model.add(Dense(
  1, activation=None,
  kernel_regularizer=regularizers.l2(0.01),
  bias_regularizer=regularizers.l2(0.01)))
model.add(Activation('sigmoid'))
 
model.compile(
  loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_history = model.fit(x_train, y_train,
  epochs=args.max_epochs, batch_size=args.batch_size,
  validation_data=(x_validate, y_validate), shuffle=True,
  callbacks=callbacks)
                          
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
#plt.ylim([0, 1])
plt.legend()
plt.show()

print('')
print('*********************************')
print('Testing best model on {} unseen data points'.format(len(x_test)))
model = load_model(args.model_path)
_, accuracy = model.evaluate(x=x_test, y=y_test)
print('Test accuracy: {:.2f}'.format(accuracy*100))

y_test_pred = model.predict(x=x_test)
y_test_pred = [a > 0.5 for a in y_test_pred]
accuracy = [a==b for a, b in zip(y_test, y_test_pred)]
moving_avg_acc = [sum(accuracy[:i]) / i for i in range(20, len(accuracy))]
plt.plot(moving_avg_acc)
plt.title('Prediction accuracy over time')
plt.xlabel('Test game (earlier to later)')
plt.ylabel('Test accuracy moving average')
plt.hlines(sum(y_test)/len(y_test), 0, len(moving_avg_acc), colors='g', label='Home team wins')
plt.hlines(0.58, 0, len(moving_avg_acc), colors='r', label='Vegas favortie wins')
plt.show()
