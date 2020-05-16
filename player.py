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
    
  def append(self, o):
    """Adds results from other object o"""
    self.batting.append(o.batting)
    self.fielding.append(o.fielding)
    self.pitching.append(o.pitching)
    
  def to_vector(self):
    return self.batting.to_vector() + self.fielding.to_vector() + self.fielding.to_vector()    
    
class FieldingStats(object):

  def __init__(self, player):
    self.NUM_FIELD_POSITIONS = 12
    self.plays_per_position = [0] * self.NUM_FIELD_POSITIONS
    self.outs_per_position = [0] * self.NUM_FIELD_POSITIONS
    self.errors_per_position = [0] * self.NUM_FIELD_POSITIONS
    self.points_per_position = [0] * self.NUM_FIELD_POSITIONS
    self._current_field_position = -1
    self.player = player
    
  def set_position(self, position):
    self._current_field_position = position
  
  def reset_position(self):
    self._current_field_position = -3
    
  def unassign_position(self, old_position):
    if old_position == self._current_field_position:
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
    return (self.plays_per_position +
      # outs and errors per play, per fielding position. Add 1 to denom to avoid zero division.
      [o / (p+1) for o, p in zip(self.outs_per_position, self.plays_per_position)] +
      [e / (p+1) for e, p in zip(self.errors_per_position, self.plays_per_position)])
    
  def append(self, o):
    """Adds results from other object o"""
    for p in range(self.NUM_FIELD_POSITIONS):
      self.plays_per_position[p] += o.plays_per_position[p]
      self.outs_per_position[p] += o.outs_per_position[p]
      self.errors_per_position[p] += o.errors_per_position[p]
      self.points_per_position[p] += o.points_per_position[p]
    
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
    
  def append(self, o):
    """Adds results from other object o"""
    for p in o.raw_pitches:
      self.raw_pitches[p] += o.raw_pitches[p]
    for r in o.results:
      self.results[r] += o.results[r]
    self.pitches_thrown += o.pitches_thrown
    self.at_bats += o.at_bats
    self.points += o.points
    self.outs += o.outs
    self.runner_advancement += 0
    
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
    at_bats_denominator = self.at_bats + 1  # Add 1 to denom to avoid zero division.
    return ([p / at_bats_denominator for p in self.raw_pitches.values()] +
            [r / at_bats_denominator for r in self.results.values()] +
            [self.pitches_thrown,
             self.at_bats,
             self.points/at_bats_denominator,
             self.outs/at_bats_denominator,
             self.runner_advancement/at_bats_denominator])
    
class BattingStats(PitchingStats):
  # Can I say battings stats just the equivalent of pitching stats, but
  # attributed to the batter rather than the pitcher?
  pass
