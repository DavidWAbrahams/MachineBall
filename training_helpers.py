from keras.callbacks.callbacks import Callback
import numpy as np

import pickle

def ShufflePlayers(samples):
  # Shuffle the order of players within each game. This prevents the model from
  # unfairly knowing who will start the game vs who will be substituted later.
  rng = np.random.default_rng()
  rng.shuffle(samples, axis=1)
  
def LoadData(sample_path, label_path, test_fraction):
  #Reads samples & labels from disk, pads them, and does training/test split.
  samples = pickle.load(open(sample_path, 'rb'))
  print('Read {} games'.format(len(samples)))
  max_num_players = max([len(s) for s in samples])
  
  # Pad all game samples to the same length
  player_len = len(samples[-1][0])
  for game in samples:
    while len(game) < max_num_players * 1.1:
      game.append([0]*player_len)
  samples = np.asarray(samples)
  ShufflePlayers(samples)
  
  labels = np.array(pickle.load(open(label_path, 'rb')))
  
  # Hold back some data for testing. It's probably import that the game
  # order has NOT been shuffled at this point, so that the test samples
  # come from chronologically later games. It may not be fair to let the
  # model train on player stats that were influenced by the test games.
  samples_train, samples_test = np.split(
    samples, [int((1-test_fraction) * len(samples))])
  labels_train, labels_test = np.split(
    labels, [int((1-test_fraction) * len(labels))])
    
  # Convert these views to copies, so that no cross-edits happen
  return (np.copy(samples_train), np.copy(samples_test),
          np.copy(labels_train),  np.copy(labels_test))
  
class ShuffleCallback(Callback): 
  """Shuffling the order of players denies the model unfair knowledge about who
  is starting, so it initially performs way worse. But if we shuffle at each
  epoch then it acts like data augmentation, allowing us to train MUCH longer
  before overfitting."""
  def __init__(self, samples):
    self.samples = samples
    
  def on_epoch_begin(self, epoch, logs={}):
    ShufflePlayers(self.samples)