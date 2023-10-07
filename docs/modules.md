# Modules

## src.utils
Utility functions. These functions are used in mutliple places throughout the source code and cannot be coupled to any particular module.

## src.data.build
Build the full set of training and testing data from scratch.

Receives a set of paths from setup.sh, then fetches raw data, builds features, and assembles training data. Then it splits holdout data by year and writes to disk.

## src.data.features.travel
Helper functions for building engineered features for team travel.

## src.data.features.team_stats
Helper functions for building engineered features for team stats.

## src.data.features.weather
Helper functions for building weather features.

## src.data.raw.elo
Helper functions for fetching raw elo ratings from 538.

## src.data.raw.plays
Helper functions for fetching raw NFL play-by-play data.

https://github.com/nflverse/nflverse-data/releases/tag/pbp

## src.data.raw.weather
Helper functions for fetching raw weather data.

## src.data.raw.games
Helper functions for fetching and saving NFL game data.

## src.data.train.target
Helper functions for building target column for games data.

## src.data.train.train
Helper functions for building the training data.

## src.docs.build
Extract docstrings from source files and build the documentation.

## src.model.evaluate
Helper functions for evaluating models.

## src.model.hyperoptimize
Helper functions for optimizing model hyperparameters.

Optimizer: hyperopt

## src.model.estimators
Build estimators to be used as the final step in a pipeline.

Baseline model: Logistic Regression
Working model: SWIFT

## src.model.pipeline
Helper functions for building scikit-learn pipelines.

The base pipeline consists of a single preprocessor applicable to any training/test dataset. All other pipeline functions should build on top of that base and return a complete pipeline with an estimator.

## src.model.train
Train and evaluate models.

## src.model.process
Helper functions for processing training data in a scikit pipeline.

Instead of creating bespoke classes that inherit from scikit, we'll rely on the FunctionTransformer to make them compatible with scikit pipelines. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

The preprocess function is reserved for model-agnostic transformations that should be done at the start of a pipeline.

## src.plot.plot
Helper functions for plotting model evaluation results.

## src.plot.style
Helpers for setting plot style and colors.

