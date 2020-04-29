from stats_tracker import StatsTracker
from game import Game

import glob
import random
import numpy as np
import os
import pickle

DATA_PATH = '.\\data\\'

def season_ongoing(season_event_file_lines):
  for team in season_event_file_lines:
    if Game.peakNextDate(team):
      return True
  return False
  
def data_from_game_files():
  year_dirs = [f.path for f in os.scandir(DATA_PATH) if f.is_dir()]
  year_dirs.sort()
  print('Years: {}'.format(year_dirs))
  
  games = []
  stats = StatsTracker()
  
  for year_dir in year_dirs:
    print('Processesing season {}'.format(year_dir))
    initial_num_games = len(games)
    season_event_file_lines = []
    
    for filename in glob.glob(os.path.join(year_dir, '*.EV*')):
      with open(filename, 'r') as f:
        #print('opened {}'.format(filename))
        season_event_file_lines.append([line.rstrip() for line in f])
          
    # TODO: this is pretty inefficient
    while season_ongoing(season_event_file_lines):
      next_game_dates = []
      for team_event_file_lines in season_event_file_lines:
        next_game_date = Game.peakNextDate(team_event_file_lines)
        next_game_dates.append(next_game_date)
      next_game_dates = np.array(next_game_dates, dtype=np.float)
        
      next_game_team_idx = np.nanargmin(next_game_dates)
      if next_game_dates[next_game_team_idx] == np.NaN:
        continue
      
      next_game_lines = season_event_file_lines[next_game_team_idx]
      # pass lines to game gobbler
      new_game = Game()
      new_game.gobble(next_game_lines, stats)
      games.append(new_game)
      #print('Finished game {} with score {}'.format(new_game.id, new_game.score))
      
    num_games = len(games)
    print('Parsed {} more games ({} total)'.format(num_games-initial_num_games, num_games))
    
  print('')
  print('***Done parsing game events***')
  print('Total games parsed: {}'.format(len(games)))
  sample_player = random.choice(list(stats.players.keys()))
  print('Example player stat: {}'.format(stats.get_player(sample_player).to_vector()))
  
  samples = []
  labels = []
  for game in games:
    sample, visitor_label, home_label = game.to_sample()
    samples.append(sample)
    labels.append([visitor_label, home_label])
  
  return samples, labels

def main():
  SAVED_SAMPLE_PATH = '.\\samples.p'
  SAVED_LABEL_PATH = '.\\labels.p'
  
  if os.path.isfile(SAVED_SAMPLE_PATH) and os.path.isfile(SAVED_LABEL_PATH):
    print('Using labeled data found at {}'.format(SAVED_SAMPLE_PATH))
    samples = pickle.load(open(SAVED_SAMPLE_PATH, 'rb'))
    labels = pickle.load(open(SAVED_LABEL_PATH, 'rb'))
  else:
    print('No saved training data found. Generating from raw game files.')
    samples, labels = data_from_game_files()
    assert samples
    assert labels
    assert len(samples) == len(labels), '{} vs {}'.format(len(samples), len(labels))
    print('Generated {} training samples'.format(len(samples)))
    # save for later reuse.
    pickle.dump(samples, open(SAVED_SAMPLE_PATH, 'wb'))
    pickle.dump(labels, open(SAVED_LABEL_PATH, 'wb'))
    

if __name__ == "__main__":
    main()

