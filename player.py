class Player(object):
  def __init__(self, id):
    self.id = id # to be hidden from model?)
    
    self.batting = BattingStats()
    self.fielding = FieldingStats(self)
    self.pitching = PitchingStats()
    
    #age?
    #contract status? :)
    
  def update(self, play):
    self.batting.update(play)
    self.fielding.update(play)
    self.pitching.update(play)
    
  def to_vector(self):
    return self.batting.to_vector() + self.fielding.to_vector() + self.fielding.to_vector()    
    
class FieldingStats(object):
  def __init__(self, player):
    self.plays_per_position = [0] * 12
    self.outs_per_position = [0] * 12
    self.errors_per_position = [0] * 12
    self.points_per_position = [0] * 12
    self._current_field_position = -1
    self.player = player
    
  def set_position(self, position):
    #print('{} goes to position {}'.format(self.player.id, position))
    self._current_field_position = position
  
  def reset_position(self):
    self._current_field_position = -3
    
  def unassign_position(self, old_position):
    if old_position == self._current_field_position:
      #print('****Unassigning {} position from {}'.format(self.player.id, self._current_field_position))
      self._current_field_position = -2
    
  def error(self):
    assert self._current_field_position > 0, self.player.id
    self.errors_per_position[self._current_field_position-1] += 1
    
  def participated(self):
    assert self._current_field_position > 0, self.player.id
    self.plays_per_position[self._current_field_position-1] += 1
    
  def update(self, play):
    assert self._current_field_position > 0, self.player.id
    self.plays_per_position[self._current_field_position-1] += 1
    self.outs_per_position[self._current_field_position-1] += play.outs
    self.points_per_position[self._current_field_position-1] += sum(play.points)
    if self._current_field_position in play.error_positions:
      self.errors_per_position[self._current_field_position-1] += 1
    
  def to_vector(self):
    return self.plays_per_position + self.outs_per_position + self.errors_per_position
    
class PitchingStats(object):

  def __init__(self):
    _PITCH_TYPES = ['+', '*', '.', '1', '2', '3', '>', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']
    self.raw_pitches = {x: 0 for x in _PITCH_TYPES}
    
    self._RESULT_TYPES = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'S', 'D', 'T', 'HR', 'W', 'HP', 'K', 'E', 'SB', 'BK', 'WP', 'FC', 'IW', 'DGR'])
    self.results = {x: 0 for x in self._RESULT_TYPES}
  
    self.pitches_thrown = 0
    self.at_bats = 0
    self.points = 0
    self.outs = 0
    self.runner_advancement = 0
    
    # todo
    # self.lefty
    
  def update(self, play):
    assert play.result
    assert play.result in self._RESULT_TYPES, play.result
    for pitch in play.pitches:
      if pitch == 'a': continue  # known bad data in TOR201908170
      if pitch not in self.raw_pitches:
        raise Exception('Unrecognized pitch {} in event {}'.format(pitch, play.raw_event))
      self.raw_pitches[pitch] += 1
      self.pitches_thrown += 1
    self.at_bats += 1
    self.points += sum(play.points)
    self.outs += play.outs
    self.runner_advancement += play.runner_advancement
    
  def to_vector(self):
    return list(self.raw_pitches.values()) + list(self.results.values()) + [self.pitches_thrown,
      self.at_bats,
      self.points,
      self.outs,
      self.runner_advancement]
    
class BattingStats(PitchingStats):
  # Can I say battings stats just the equivalent of pitching stats, but
  # attributed to the batter rather than the pitcher?
  pass
