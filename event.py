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
    event.parts = line.split(',') # TODO: improve this
    event.type = event.parts[0]
    return event
    
   