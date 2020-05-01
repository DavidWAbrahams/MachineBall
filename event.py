import re

class Event(object):
  class Types(object):
    id = 'id'
    info = 'info'
    start = 'start'
    sub = 'sub'
    play = 'play'

  def __init__(self):
    self.type = None
    parts = []
    self.raw = None
    
  @classmethod
  def from_line(cls, line):
    event = Event()
    event.raw = line
    # These are rare characters that just mark uncertain plays.
    line = re.sub('[#!?]', '', line)
    # TODO: improve this. For example, handle comma inside quotation marks correctly.
    event.parts = line.split(',')
    event.type = event.parts[0]
    return event
    
   