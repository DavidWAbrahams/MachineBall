from event import Event
import copy
import numpy as np

class Game(object):
  def __init__(self):
    self.id = 0
    self.date = 0
    # [visiting team, home team]
    self.teams = [None, None]
    self.score = [0, 0]
    self.initial_roster = [[], []]
    self.players = [set(), set()]
    # [ team 1 map of position:playerid, team 2 map of position:playerid]
    self.active_players = [{}, {}]
    
    
  def to_sample(self):
    sample = self.initial_roster[0] + self.initial_roster[1]
    print(np.array(sample).shape)
    return sample, self.score[0], self.score[1]
    
  @classmethod
  def peakNextDate(cls, lines):
    for line in lines:
      event_line = Event.from_line(line)
      if event_line.type == Event.Types.id:
        return int(event_line.parts[1][3:])
    return None
    
  def gobble(self, lines, stats_tracker):
    initial_stats_tracker = copy.deepcopy(stats_tracker)
  
    # consumes event lines until the game appears to be over
    if not self.id:
      line = lines.pop(0)
      id_event = Event.from_line(line)
      assert id_event.type == Event.Types.id, id_event.type
      self.id = id_event.parts[1]
      self.date = int(self.id[3:])
      date_prefix = str(self.date)[:2]
      assert date_prefix in ['19', '20'], date_prefix # sanity check that year is 19xx or 20xx 
      #print('Starting game {}'.format(self.id))

    while lines:
      # If we've reached another game, reset all current player positions
      # and don't consume the line.
      line = lines[0]
      new_event = Event.from_line(line)
      if new_event.type == Event.Types.id:
        stats_tracker.reset_player_positions()
        break
        
      lines.pop(0)
      
      # note home and away teams
      if new_event.type == Event.Types.info:
        if new_event.raw[1] == 'visteam':
          self.teams[0] = new_event.raw[2]
        elif new_event.raw[1] == 'hometeam':
          self.teams[1] = new_event.raw[2]
          
      # track the currently active players
      elif new_event.type in [Event.Types.start, Event.Types.sub]:
        # todo: i dunno if this works for designated hitters, pinch hitters, pinch runners
        _, player_id, _, team, _, position = new_event.parts
        team, position = int(team), int(position)
        self.players[team].add(player_id)
        if position in self.active_players[team]:
          # the old player needs to be unassigned IF they aren't already
          # in another position.
          stats_tracker.unassign_player(player_id=self.active_players[team][position], old_position=position)
        self.active_players[team][position] = player_id
        stats_tracker.set_player_position(player_id, position)
        
      
      elif new_event.type == Event.Types.play:
        # update score
        score_update = stats_tracker.play(new_event, batter_id=player_id, fielder_ids=self.active_players[team])
        self.score = [sum(x) for x in zip(self.score, score_update)]
        
        
    # Now that we know which players actually saw field time, go back
    # and get their initial stats (before this game was played). This is
    # the training data.
    # If we had no record of a player before this game, then sadly
    # we can't include them.
    for team in [0, 1]:
      for player_id in self.players[team]:
        if initial_stats_tracker.has_player(player_id):
          player_vector = initial_stats_tracker.get_player(player_id).to_vector()
          player_vector.append(team)  # mark visitor/home
          self.initial_roster[team].append(player_vector)