# MachineBall

Use public baseball data to train machine learning models. Baseball has just about the richest data set of any sport. Thousands of games are played per year and each one is [documented play-by play for free](https://www.retrosheet.org/game.htm) going back to 1919. But the data are trapped in a [fairly painful file format](https://www.retrosheet.org/eventfile.htm). For example, can you (or your ML project) understand the input "play,4,0,harpb001,22,BBFSFX,FC5/G5.3XH(52)"? This project converts the data into a format that you may find more useful for machine learning purposes. Specifically, it parses each game into a training sample that consists of:
1. A list containing the stats of every player in the game (snapshotted before the game began). Each player is marked visiting team (0) or home team (1).
2. The final score of the game.

An example Keras model is provided that achieves ~59% accuracy predicting game winners (if told the starters for that game) (on heldout test data) (given 30 years of training data). This compares favorably with [Vegas odds, which pick the winning team ~58% of the time](https://www.oddsshark.com/sports-betting/which-sport-do-betting-underdogs-win-most-often). This accuracy applies to ~300 games immediately after the training data. Accuracy degrades to ~53% if you try a prediction 600 games later.

## Words of Warning

I know almost nothing about baseball, and it shows. Baseball has an huge dictionary of vital player stats that I totally ignored. What I've mostly done is parse each play and then try to credit/penalize specific players based on how many points, outs, errors, and bases were gained. Batters and pitchers are credited in each pitch, hit, and play. Fielders are credited in a play if they are mentioned specifically. Fielding stats are recorded per-position (so a player may have different stats when playing 1st base vs 3rd).
I'm SURE there are serious parsing bugs due to my clumsy code, the confusing file format, and my misunderstanding of baseball. I'm hoping it generates something useful for ML training anyway.

## Getting Started

### Prerequisites

Game parsing requires just:
```
pip install numpy
```

The Keras model requires TensorFlow, Keras, and
```
pip install h5py
pip install matplotlib
```

### Parsing game logs
Download some number of Retrosheet [game EVENT files](https://www.retrosheet.org/game.htm) (NOT box score files) and unzip them in subdirectories under 'data'. For example your directory structure could look like this:

machineball\parse.py

machineball\data\2018ev\2018ANA.EVA

machineball\data\2018ev\2018ARI.EVN

...

machineball\data\2019ev\2019ANA.EVA

machineball\data\2019ev\2019ARI.EVN

...

It is important that the directories under 'data' are named so that they sort chronologically. For instance, 'data\2018ev' and 'data\2019ev' is fine. But 'data\b2018' and 'data\a2019' is not, since a2019 would be erroneously parsed before b2018. The Retrosheet zip files are named appropriately so no changes should be needed.

The decade-length Retrosheet event files are fine too, but they are going to take a long time to parse.

Once you've placed the game logs, you can begin parsing training samples. Simply run
```
python parse.py --data_path=data --parsed_data_prefix=<name your data here> --roster_style=participants
```

This takes a few minutes per season. Once finished, the results are saved as Python pickles for fast reuse. If you want to regenerate the samples later with more data, just delete the pickle files (samples.p and labels.p in your app dir) and rerun the app.
If you get memory exceptions from trying to parse too much, use "--max_pickle_len=10000" to split data into multiple output files.

There are a number of ways the system can guess the team roster of each game.
* Read ahead and see everyone who participates (--roster_style=participants)
* Read ahead and see who actually starts in this game (--roster_style=starters)
* Use the players who participated in the team's last game (--roster_style=last)
* Use the team's entire roster, whether or not they participate in this game (--roster_style=full)

Some of these are possible to guesse before a game starts, some are not. (Ones that are not guessable before a game would be useless for prediction.) Some train better, some train worse.

### Training a model
I've provided an example Keras model that predicts which team will win, given the stats of all players participating. Simply run:
```
python keras_winner.py --parsed_data_prefix=<name your data here>
```
This will read the previously generated data and train a multi-layer bidirectional LSTM to predict game winners. The model is saved to disk for later use. It achieves ~65% accuracy when trained on all games from 2017-2019

Another model predicts the point spread of a game.
```
python keras_spread.py
```
Its test predictions are wrong by a median of about 2.5 points when trained on all games from 2010-2019.

## Authors

* **David Abrahams** - [AllWashedOut](https://github.com/AllWashedOut)

## Acknowledgments

* Retrosheet is crazy, man. Unimaginable dedication.
* Moneyball was a pretty good book even for someone who doesn't know jack about sports.
