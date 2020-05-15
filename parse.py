from stats_tracker import StatsTracker
from game import Game

import argparse
import glob
import numpy as np
import os
import pickle
import random

parser = argparse.ArgumentParser()

parser.add_argument('--sample_path', action='store', default='.\\samples.p', dest='sample_path',
                    help='Output path for training sample pickle')
parser.add_argument('--label_path', action='store', default='.\\labels.p', dest='label_path',
                    help='Output path for training label pickle')
parser.add_argument('--data_path', action='store', default='.\\data\\', dest='data_path',
                    help='Input data dir to parse')
parser.add_argument('--starters', action='store', default=False, dest='starters_only',
                    help='Only train on starting players, not substitutes.')               

args = parser.parse_args()

def season_ongoing(season_event_file_lines):
  for team in season_event_file_lines:
    if Game.peakNextDate(team):
      return True
  return False
  
def data_from_game_files():
  year_dirs = [f.path for f in os.scandir(args.data_path) if f.is_dir()]
  year_dirs.sort()
  print('Years: {}'.format(year_dirs))
  
  games = []
  stats = StatsTracker()
  
  for year_dir in year_dirs:
    print('Processesing season {}'.format(year_dir))
    initial_num_games = len(games)
    season_event_file_lines = []
    
    # read all games from this season to RAM
    for filename in glob.glob(os.path.join(year_dir, '*.EV*')):
      with open(filename, 'r') as f:
        season_event_file_lines.append([line.rstrip() for line in f])
          
    # Parse the season's games in chronological order
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
      #print('Finished parsing game {} with score {}'.format(new_game.id, new_game.score))
      
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
  
  if os.path.isfile(args.sample_path) or os.path.isfile(args.label_path):
    print('ERROR: Parsed game data already exists. Please delete {} and {} if you are intentionally recreating it.'.format(args.sample_path, args.label_path))
  else:
    print('No saved training data found. Generating from raw game files.')
    samples, labels = data_from_game_files()
    assert samples
    assert labels
    assert len(samples) == len(labels), '{} vs {}'.format(len(samples), len(labels))
    print('Generated {} training samples'.format(len(samples)))
    # save for later model training.
    pickle.dump(samples, open(args.sample_path, 'wb'))
    pickle.dump(labels, open(args.label_path, 'wb'))

if __name__ == "__main__":
    main()

