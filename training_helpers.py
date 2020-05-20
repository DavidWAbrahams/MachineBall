from keras.callbacks.callbacks import Callback
import numpy as np

import argparse
import glob
import pickle

def ShufflePlayers(samples):
  # Shuffle the order of players within each game. This prevents the model from
  # unfairly knowing who will start the game vs who will be substituted later.
  rng = np.random.default_rng()
  rng.shuffle(samples, axis=1)
  
def FindCenter(sample):
  for i, player in enumerate(sample):
    if player[-1]:
      return i
    #print('Found no home team players?')
    return 0
  
def LoadData(parsed_data_prefix, drop_fraction=0, validate_fraction=0.1, test_fraction=0.1):
  #Reads samples & labels from disk, pads them, and does training/test split.
  samples = []
  labels = []
  for filename in glob.glob(parsed_data_prefix + '_samples*.*'):
    samples.extend(pickle.load(open(filename, 'rb')))
  for filename in glob.glob(parsed_data_prefix + '_labels*.*'):
    labels.extend(pickle.load(open(filename, 'rb')))
  print('Read {} games'.format(len(samples)))
  max_num_players = max([len(s) for s in samples])
  
  nonempty_mask = [bool(sample[-1][-1]) for sample in samples]
  print('Dumped {} empty samples'.format(len(nonempty_mask)-sum(nonempty_mask)))
  samples = [b for a, b in zip(nonempty_mask, samples) if a]
  labels = [b for a, b in zip(nonempty_mask, labels) if a]
  
  # Pad all game samples to the same length
  player_len = 0
  for sample in samples:
    if sample:
      player_len = max(player_len, len(sample[0]))
  print('Players have {} stats each'.format(player_len))
  assert player_len
  for game in samples:    
    center = FindCenter(game)
    while len(game) < max_num_players * 1.1:
      game.insert(center, [0]*player_len)
  samples = np.asarray(samples)
  labels = np.array(labels)
  
  # Drops the first few samples, on the theory that you may want to
  # train on just later samples when there is more player data.
  samples = samples[int(len(samples)*drop_fraction):]
  labels = labels[int(len(labels)*drop_fraction):]
  
  # Hold back some data for testing. It's probably import that the game
  # order has NOT been shuffled at this point, so that the test samples
  # come from chronologically later games than training. It may not be
  # fair to let the model train on player stats that were influenced by
  # the test games.
  samples_train, samples_test, samples_validate = np.split(
    samples, [int((1-test_fraction-validate_fraction) * len(samples)), int((1-validate_fraction) * len(samples))])
  labels_train, labels_test, labels_validate = np.split(
    labels, [int((1-test_fraction-validate_fraction) * len(labels)), int((1-validate_fraction) * len(labels))])
    
  assert len(samples_train) + len(samples_validate) + len(samples_test) == len(samples)
  
  print('{} train, {} validate, {} test samples.'.format(len(samples_train), len(samples_validate), len(samples_test)))
  print('Home team won {:.1f}% of training games and {:.1f}% of test games. Your model better beat this.'.format(
    HomeTeamWinRate(labels_train)*100,
    HomeTeamWinRate(labels_test)*100))
    
  # Convert these views to copies, so that no cross-edits happen
  return (np.copy(samples_train), np.copy(samples_validate), np.copy(samples_test),
          np.copy(labels_train),  np.copy(labels_validate),  np.copy(labels_test))
          
def HomeTeamWinRate(labels):
  winners = [label[1] > label[0] for label in labels]
  return sum(winners)/len(winners)
  
class ShuffleCallback(Callback): 
  """Shuffling the order of players denies the model unfair knowledge about who
  is starting, so it initially performs way worse. But if we shuffle at each
  epoch then it acts like data augmentation, allowing us to train MUCH longer
  before overfitting."""
  def __init__(self, samples):
    self.samples = samples
    
  def on_epoch_begin(self, epoch, logs={}):
    ShufflePlayers(self.samples)
    
def TrainingArgs():
  parser = argparse.ArgumentParser()

  parser.add_argument('--parsed_data_prefix', action='store', default='.\\out', dest='parsed_data_prefix',
                      help='Path prefix for training data pickles')
  parser.add_argument('--model_path', action='store',
                      default='winner_model.h5', dest='model_path',
                      help='Output path for the saved model')
  parser.add_argument('--max_epochs', action='store',
                      default=600, dest='max_epochs',
                      help='Max training epochs', type=int)
  parser.add_argument('--batch_size', action='store',
                      default=1000, dest='batch_size',
                      help='Max training epochs', type=int)
  parser.add_argument('--patience', action='store',
                      default=50, dest='patience',
                      help='Training patience', type=int)
  parser.add_argument('--rnn_layers', action='store',
                      default=3, dest='num_rnn_layers',
                      help='Training patience', type=int)
  parser.add_argument('--rnn_layer_size', action='store',
                      default=128, dest='rnn_layer_size',
                      help='Training patience', type=int)
  parser.add_argument('--roster_shuffle', action='store_true',
                      default=False, dest='roster_shuffle',
                      help='Shuffle player order on rosters each epoch.')
  parser.add_argument('--validate_fraction', action='store',
                      default=0.04, dest='validate_fraction',
                      help='holdout validate data fraction', type=float)
  parser.add_argument('--test_fraction', action='store',
                      default=0.04, dest='test_fraction',
                      help='holdout test data fraction', type=float)
                      
  return parser.parse_args()