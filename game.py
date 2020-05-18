from event import Event
from stats_tracker import StatsTracker
import copy
import numpy as np

class Game(object):
  def __init__(self):
    self.id = 0
    self.date = 0
    self.year = 0
    # [visiting team, home team]
    self.team_ids = [None, None]
    self.score = [0, 0]
    self.initial_full_roster = [[], []]
    self.initial_starting_roster = [[], []]
    self.player_ids = [set(), set()]
    self.starting_player_ids = [set(), set()]
    # [ team 1 map of position:playerid, team 2 map of position:playerid]
    self.active_players = [{}, {}]
    
    self._last_event_type = None
    
  @classmethod
  def peakNextDate(cls, lines):
    """Given some lines from an event file, finds the date of the next game
    in the lines."""
    for line in lines:
      event_line = Event.from_line(line)
      if event_line.type == Event.Types.id:
        return int(event_line.parts[1][3:])
    return None
    
  def gobble(self, lines, persistent_stats_tracker, roster_style='participants', full_rosters=None, last_game_rosters=None):
    """Given some lines from an event file, reads the plays for one game.
    The lines are consumed, so you can call this repeatedly on a list of
    events to parse out all the games."""
    
    # Note that persistent_stats_tracker shouldn't be touched until the end of the method,
    # since some code assumes this is pristine from before the game started.
    # Instead use a temporary per-game tracker and then merge them at the end.
    game_stats_tracker = StatsTracker()
      
    # consumes event lines until the game appears to be over
    while not self.id:
      line = lines.pop(0)
      id_event = Event.from_line(line)
      if id_event.type == Event.Types.id:
        self.id = id_event.parts[1]
        print('Parsing game {}'.format(self.id), end='\r')
        self.date = int(self.id[3:])
        date_prefix = str(self.date)[:1]
        assert date_prefix in ['1', '2'], date_prefix # sanity check that year is like 19xx or 2xxx. TODO fix in 1k years.
        self.year = str(self.date)[:4]
      else:
        print('Skipping line: {}'.format(line))

    while lines:
      # If we've reached another game, don't consume the line.
      line = lines[0]
      new_event = Event.from_line(line)
      if new_event.type == Event.Types.id:
        break
        
      lines.pop(0)
      
      if (self._last_event_type == Event.Types.start and
          new_event.type != Event.Types.start):
        for team in [0, 1]:
          team_roster = full_rosters[self.year][self.team_ids[team]]
          # record players for 'starters only' roster training
          if roster_style == 'starters':
            for player_id in self.player_ids[team]:
              player_vector = self.player_vector(player_id, team, persistent_stats_tracker, team_roster)
              self.initial_starting_roster[team].append(player_vector)
          if roster_style in ['full', 'last']:
            if roster_style == 'last':
              # Filter the roster list to only include players who participated in the last game
              team_roster = dict(filter(lambda elem: elem[0] in last_game_rosters[self.team_ids[team]], team_roster.items()))
            for player_id in team_roster:
              player_vector = self.player_vector(player_id, team, persistent_stats_tracker, team_roster)
              self.initial_full_roster[team].append(player_vector)
      # note home and away teams
      if new_event.type == Event.Types.info:
        if new_event.parts[1] == 'visteam':
          self.team_ids[0] = new_event.parts[2]
        elif new_event.parts[1] == 'hometeam':
          self.team_ids[1] = new_event.parts[2]
          
      # track the currently active players
      elif new_event.type in [Event.Types.start, Event.Types.sub]:
        # todo: i dunno if this works for designated hitters, pinch hitters, pinch runners
        _, player_id, _, team, _, position = new_event.parts
        team, position = int(team), int(position)
        self.player_ids[team].add(player_id)
        if new_event.type == Event.Types.start:
          self.starting_player_ids[team].add(player_id)
        if position in self.active_players[team]:
          # the old player needs to be unassigned IF they aren't already
          # in another position.
          game_stats_tracker.unassign_player(player_id=self.active_players[team][position], old_position=position)
        self.active_players[team][position] = player_id
        game_stats_tracker.set_player_position(player_id, position)
      elif new_event.type == Event.Types.play:
        # process play and update score
        score_update = game_stats_tracker.play(new_event, batter_id=player_id, fielder_ids=self.active_players[team])
        self.score = [sum(x) for x in zip(self.score, score_update)]
        
      self._last_event_type = new_event.type
    
    # Now that we know which players actually saw field time, go back
    # and get their initial stats (before this game was played). This is
    # the training data.
    # If we had no record of a player before this game, then sadly
    # we can't include them.
    if roster_style == 'participants':
      for team in [0, 1]:
        team_roster = full_rosters[self.year][self.team_ids[team]]
        for player_id in self.player_ids[team]:
          player_vector = self.player_vector(player_id, team, persistent_stats_tracker, team_roster)
          self.initial_full_roster[team].append(player_vector)
            
    persistent_stats_tracker.append(game_stats_tracker)
    
  def player_vector(self, player_id, home_or_visitor, stats_tracker, team_roster):
    if stats_tracker.has_player(player_id):
      player_vector = stats_tracker.get_player(player_id).to_vector()
    else:
      player_vector = Player(player_id).to_vector()
    player_vector.append(home_or_visitor)  # mark visitor/home
    player_vector.append(ord(team_roster[player_id]['batting_hand']))  # mark batting hand
    player_vector.append(ord(team_roster[player_id]['throwing_hand']))  # mark throwing hand
    player_vector.append(int(player_id in last_game_rosters[self.team_ids[home_or_visitor]])) # mark 1 if player played last game
    return player_vector
    
  def participant_ids(self):
    # useful for the 'last' roster strategy, where we assume the coach will play the same
    # players as the last game.
    return self.year, self.team_ids[0], self.player_ids[0], self.team_ids[1], self.player_ids[1]
          
  def to_sample(self, starters_only=False):
    """Called after a game has been parsed, returns the initial stats of all
    players and the final score.
    
    Params
      starters_only: train only on the list of starting players, discarding info on subs."""
    assert self.id, 'It appears this game has not been populated. Cannot sample it.'      
     
    if starters_only:
      sample = self.initial_starting_roster[0] + self.initial_starting_roster[1]
    else:
      sample = self.initial_full_roster[0] + self.initial_full_roster[1]
    return sample, self.score[0], self.score[1]