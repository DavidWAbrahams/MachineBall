from stats_tracker import StatsTracker
from game import Game

import argparse
from collections import defaultdict
from collections import OrderedDict
import glob
import numpy as np
import os
import pickle
import os

parser = argparse.ArgumentParser()

parser.add_argument('--parsed_data_prefix', action='store', default='.\\out', dest='parsed_data_prefix',
                    help='Output path for training sample pickles')
parser.add_argument('--data_path', action='store', default='.\\data\\', dest='data_path',
                    help='Input data dir to parse')
parser.add_argument('--roster_style', action='store', default='participants', dest='roster_style', choices=['starters', 'participants', 'full', 'last'],
                    help='How to populate the roster of each team.')      
parser.add_argument('--f', action='store_true', default=False, dest='force',
                    help='Force overwrite of existing data.')
parser.add_argument('--max_pickle_len', action='store', default=50000, dest='max_pickle_len',
                    help='Max entries per pickle. May result in multiple pickles.', type=int)
parser.add_argument('--float_precision', action='store_true', default=False, dest='float_precision',
                    help='Max entries per pickle. May result in multiple pickles.')                     

args = parser.parse_args()

def season_ongoing(season_event_file_lines):
  # checks if there are any unparsed games left in the data files
  for team in season_event_file_lines:
    if Game.peakNextDate(team):
      return True
  return False
  
def data_from_roster_files():
  # Read the annual lineup for every team, found in .ROS data files
  year_dirs = [f.path for f in os.scandir(args.data_path) if f.is_dir()]
  year_dirs.sort()
  rosters = OrderedDict() # year: team: [player1, player2, ...]
  
  for year_dir in year_dirs:
    print('Processesing season {} rosters'.format(year_dir))
    season_event_file_lines = []
    
    # read all rosters from this season to RAM
    for filename in glob.glob(os.path.join(year_dir, '*.ROS*')):
      with open(filename, 'r') as f:
        roster_name = os.path.splitext(os.path.basename(filename))[0]
        year = roster_name[-4:]
        assert len(year) == 4
        assert year[:2] in ['19', '20']
        team = roster_name[0:-4]
        if year not in rosters:
          rosters[year] = {}
        if team not in rosters[year]:
          rosters[year][team] = OrderedDict()
        
        line_parts = [line.rstrip().split(',') for line in f]
        for player_parts in line_parts:
          player_id = player_parts[0]
          batting_hand = player_parts[3]
          throwing_hand = player_parts[4]
          rosters[year][team][player_id] = {'batting_hand': batting_hand, 'throwing_hand': throwing_hand}
        
  return rosters
  
def data_from_game_files():
  # Read all games from data files.
  year_dirs = [f.path for f in os.scandir(args.data_path) if f.is_dir()]
  year_dirs.sort()
  print('Years: {}'.format(year_dirs))
  
  games = []
  stats = StatsTracker()
  
  full_rosters = data_from_roster_files()
  last_game_rosters = defaultdict(dict)
  
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
      new_game = Game(float_precision=args.float_precision)
      new_game.gobble(next_game_lines, stats, roster_style=args.roster_style, full_rosters=full_rosters, last_game_rosters=last_game_rosters)
      games.append(new_game)
      #print('Finished parsing game {} with score {}'.format(new_game.id, new_game.score))
      # track players for each team for the 'last' roster strategy
      _, visitor_team, visitor_ids, home_team, home_ids = new_game.participant_ids()
      last_game_rosters[visitor_team] = visitor_ids
      last_game_rosters[home_team] = home_ids
      
    num_games = len(games)
    print('Parsed {} more games ({} total)'.format(num_games-initial_num_games, num_games))
    
  print('')
  print('***Done parsing game events***')
  print('Total games parsed: {}'.format(len(games)))
  
  samples = []
  labels = []
  game_ids = []
  for game in games:
    if game.is_good_sample():
      sample, visitor_label, home_label = game.to_sample(starters_only=args.roster_style=='starters')
      samples.append(sample)
      labels.append([visitor_label, home_label])
      game_ids.append(game.id)
      
  print('Purged {} out of {} games due to sparse player stats.'.format(len(games)-len(samples), len(games)))
    
  print('Example player stats:')
  print(samples[-1][-1])
  
  return samples, labels, game_ids

def main():
  
  if not args.force and (os.path.isfile(args.sample_path) or os.path.isfile(args.label_path)):
    print('ERROR: Parsed game data already exists. Please use --f if you are intentionally recreating it.')
  else:
    print('No saved training data found. Generating from raw game files.')
    samples, labels, game_ids = data_from_game_files()
    assert samples
    assert labels
    assert len(samples) == len(labels), '{} vs {}'.format(len(samples), len(labels))
    print('Generated {} training samples'.format(len(samples)))
    # save for later model training.
    for i in range(int(len(labels)/args.max_pickle_len) + 1):
      # Save data in chunks to avoid OOM errors during pickling.
      start = i*args.max_pickle_len
      end = (i+1)*args.max_pickle_len
      pickle.dump(labels[start:end], open(args.parsed_data_prefix + '_labels_{}.p'.format(i), 'wb'))
      pickle.dump(samples[start:end], open(args.parsed_data_prefix + '_samples_{}.p'.format(i), 'wb'))
      pickle.dump(game_ids[start:end], open(args.parsed_data_prefix + '_gameids_{}.p'.format(i), 'wb'))

if __name__ == "__main__":
    main()

