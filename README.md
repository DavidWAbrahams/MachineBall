# MachineBall

Use public baseball data to train machine learning models. Baseball has one of the richest data sets of any sport in history. Thousands of games are played per year and each one is [documented play-by play for free](https://www.retrosheet.org/game.htm) going back to 1919. But the data are trapped in a [fairly painful file format](https://www.retrosheet.org/eventfile.htm). For example, can you (or your ML project) understand the input "play,4,0,harpb001,22,BBFSFX,FC5/G5.3XH(52)"? This project converts the data into a format that you may find more useful for machine learning purposes. Specifically, it parses each game into a training sample that consists of:
1. A list containing the stats of every player in the game (snapshotted before the game began). Each player is marked visiting team (0) or home team (1).
2. The final score of the game in the form [visiting team score, home team score]

## Words of Warning

I know almost nothing about baseball, and it definitely shows. Baseball has an huge dictionary of vital player stats that I totally ignored. What I've mostly done is track each play and then try to credit/penialize specific players based on how many points, outs, errors, and bases were gained. Batters and pitchers are credited in each pitch, hit, and play. Fielders are credited in a play if they are mentioned specifically. Fielding stats are recorded per-position (so a player may have different stats when playing 1st base vs 3rd).
I'm SURE there are serious bugs in my parsing due to my clumsy code, the confusing file format, and my misunderstanding of baseball. I'm hoping it generates something useful for ML training anyway.

## Getting Started

Download some number of Retrosheet game event files and unzip them in subdirectories under 'data'. For example your directory structure could look like this:
machineball\main.py
machineball\data\2018ev\2018ANA.EVA
machineball\data\2018ev\2018ARI.EVN
...
machineball\data\2019ev\2019ANA.EVA
machineball\data\2019ev\2019ARI.EVN
...

It is important that the directories under 'data' are named so that they sort chronologically. For instance, 'data\2018ev' and 'data\2019ev' is fine. But 'data\b2018' and 'data\a2019' is not, since a2019 would be erroneously parsed before b2018.

Once you have done this, you can begin parsing games into training samples. Simply run
```
python main.py
```

This takes a few minutes per season. Once finished, the results are saved as Python pickles for fast reuse. If you want to regenerate the samples later with more data, just delete the pickle files (samples.p and labels.p in your app dir) and rerun the app.

### Prerequisites

```
pip install numpy
```

## Authors

* **David Abrahams** - *Initial work* - [AllWashedOut](https://github.com/AllWashedOut)

## Acknowledgments

* Retrosheet is crazy, man. Unimaginable dedication.
* Moneyball was a pretty good book even for someone who doesn't know jack about sports.
