from event import Event
from play import Play
from player import Player

from collections import defaultdict

class StatsTracker(object):
  def __init__(self):
    self.players = {}
    
  def get_player(self, player_id):
    return self.players[player_id]
    
  def has_player(self, player_id):
    return player_id in self.players
    
  def _getOrCreate(self, player_id):
    if player_id not in self.players:
      self.players[player_id] = Player(player_id)
    return self.players[player_id]
    
  def set_player_position(self, player_id, position):
    self._getOrCreate(player_id).fielding.set_position(position)
  
  def unassign_player(self, player_id, old_position):
    # the old player needs to be unassigned IF they aren't already
    # in another position.
    self.players[player_id].fielding.unassign_position(old_position)
    
  def reset_player_positions(self):
    for player in self.players.values():
      player.fielding.reset_position()
    
  def play(self, play_event, batter_id, fielder_ids):
    new_play = Play.from_event(play_event)
    
    pitcher_id = fielder_ids[1]
    catcher_id = fielder_ids[2]
    if new_play.result == 'NP':
      # NP = No play, a place holder
      return [0, 0]
    if new_play.result == 'PB':
      # PB = passed ball, a catcher error
      self._getOrCreate(catcher_id).fielding.participated()
      self._getOrCreate(catcher_id).fielding.error()
    elif new_play.result in ['CS', 'CSH', 'PO', 'POCS', 'POCSH', 'FLE', 'OA', 'SB', 'SBH', 'DI', 'C']:
      # Things that don't automatically involve the pitcher and catcher.
      # caught stealing, picked off, error on a foul ball, other advance, stolen bases, defensive indifference, interference
      # TODO this could definitely have better attribution.
      pass
    else:
      self._getOrCreate(pitcher_id).pitching.update(new_play)
      self._getOrCreate(batter_id).batting.update(new_play)
    
    for fielder_position in new_play.fielders_involved | new_play.error_positions:
      fielder_id = fielder_ids[fielder_position]
      self._getOrCreate(fielder_id).fielding.update(new_play)
    
    return new_play.points