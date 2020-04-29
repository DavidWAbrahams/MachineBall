from event import Event

import re

class Play(object):

  def __init__(self):
    self.result = None
    self.error_positions = set()
    self.fielders_involved = set()
    self.points = [0, 0]  # away points, home points
    self.outs = 0
    self.runner_advancement = 0
    self.pitches = []
    self.team_at_bat = -1  # 0 for visitors 1 for home
    self.raw_event = None
    
  def _parenthetical_field_pos(self):
    fielders = re.findall('([0-9])', self.result)
    if fielders:
      return set(int(x) for x in fielders[0].strip('()'))
    else:
      return set()
    
  @classmethod
  def from_event(cls, play_event):
    new_play = Play()
    new_play.raw_event = play_event
  
    type, inning, team_at_bat, player_at_bat, _, pitches, play = play_event.parts
    assert type == Event.Types.play
    new_play.team_at_bat = int(team_at_bat)
    assert new_play.team_at_bat in [0, 1], new_play.team_at_bat
    
    play_components = play.split('.')
    assert len(play_components) in [1, 2], play_components
    
    #todo parse hit type and fielding
    play_details = play_components[0].split('/')
    basic_play = play_details[0]
    modifiers = play_details[1:]
    if basic_play[0].isalpha():
      # starts with letters, optionally followed by num: on base, fielded by num
      new_play.result = re.findall('^[a-zA-Z]*', basic_play)[0]
      if new_play.result in ['E', 'FLE']:
        # track who made an error
        error_positions = re.findall('E[0-9]*', basic_play)
        for error_pos in error_positions:
          for error in error_pos:
            if error.isdigit():
              new_play.error_positions.add(int(error))
      elif new_play.result == 'C':
        # track who made interference.
        # Format is "C/E#" (where # is field position)
        new_play.error_positions.add(int(play_details[1][1]))
      elif new_play.result[0] == 'K':
        new_play.outs += 1
        if re.findall('[0-9]{4}', new_play.result):
          new_play.outs += 2
        elif re.findall('[0-9]{2}', new_play.result):
          new_play.outs += 1
      elif new_play.result in ['H', 'HR']:
        # Home run
        new_play.points[new_play.team_at_bat] += 1
      elif new_play.result.startswith('SB'):
        # TODO: stolen base
        new_play.runner_advancement += 1
      elif new_play.result.startswith('CS'):
        if 'E' not in new_play.result:
          # TODO: better tracking of the error
          new_play.outs += 1
        new_play.fielders_involved = new_play._parenthetical_field_pos()
      elif new_play.result.startswith('PO'):
        # includes PO (picked off) and POCS (picked off caught stealing)
        new_play.outs += 1
        new_play.fielders_involved = new_play._parenthetical_field_pos()
      elif new_play.result.startswith('FLE'):
        # error on foul ball
        fielders = re.findall('[0-9]', new_play.result)
        if fielders:
          new_play.error_positions = set(int(x) for x in fielders[0].strip('()'))
      else:
        #print(basic_play)
        pass
        # raise Exception('event: {}, play: {}, play details: {}, basic play: {}, modifiers: {}'.format(play_event, play, play_details, basic_play, modifiers))
      if basic_play[-1].isdigit():
        # track the fielders involved in the play
        # Remove parts in parenthesis, they're for doubles or something.
        # TODO: handle doubles etc.
        new_play.outs += 1
        basic_play = re.sub('\(.*\)', ' ', basic_play)
        new_play.fielders_involved = re.findall('[0-9]', basic_play)
        new_play.fielders_involved = set([int(pos) for pos in new_play.fielders_involved]) 
    elif basic_play[0].isdigit():
      # starts with a number: out by those fielders
      new_play.fielders_involved = re.findall('^[0-9]*', basic_play)[0]
      new_play.fielders_involved = list(new_play.fielders_involved)
      new_play.fielders_involved = [int(pos) for pos in new_play.fielders_involved]
      new_play.result = new_play.fielders_involved[-1]
      new_play.fielders_involved = set(new_play.fielders_involved)
    else: raise Exception(basic_play)
    
    if len(play_components) > 1:
      # handle the runners rounding bases, aka advances.
      advances = play_components[1].split(';')
      for advance in advances:
        # pull out the parenthesis, where fielders are credited
        fielding_regex = '\(.*\)'
        fielders = re.findall(fielding_regex, advance)
        if fielders:
          for i, fielder in enumerate(fielders[0]):
            if i > 0 and fielders[0][i-1] == 'E':
              # credit an error if prev char was E
              new_play.error_positions.add(int(fielder))
            elif fielder.isdigit():
              new_play.fielders_involved.add(int(fielder))
          
        advance = re.sub(fielding_regex, '', advance)
      
        # advances look like [1-H] or [2x3] etc.
        assert re.match('[0-3B][-X][0-3H]', advance), advance
        if re.search('-H', advance):
          new_play.points[new_play.team_at_bat] += 1
        if advance in ['1-2', '2-3', '3-H']:
          new_play.runner_advancement += 1
        elif advance in ['1-3', '2-H']:
          new_play.runner_advancement += 2
        elif advance in ['1-H']:
          new_play.runner_advancement += 3
        elif advance in ['1-1', '2-2', '3-3']:
          pass
        elif re.search('X', advance):
          new_play.outs += 1
        elif re.search('B-', advance):
          # TODO. indicates the batter should have been out but for a fielding error.
          pass
        else:
          raise Exception(advance)

    new_play.pitches = list(pitches)
    
    # Simplify the result to just the leading letters or single leading number.
    # TODO: leaving the numbers might make the data richer. Like, a D9 is a
    # double hit to a specific location.
    assert new_play.result
    new_play.result = str(new_play.result)
    new_play.result = re.findall('^[A-Z]*', new_play.result)[0] or re.findall('^[0-9]', new_play.result)[0]
    
    return new_play
    
   