# Modules

## src.utils
Utility functions. These functions are used in mutliple places throughout the source code and cannot be coupled to any particular module.

## src.__init__
None

## src.data.build
Builds the full set of training data from scratch.

Receives a set of paths from setup.sh, then fetches raw data, builds features, and assembles training data.


## src.data.features.travel
Helper functions for building engineered features for team travel.

## src.data.features.weather
Helper functions for building weather features.

## src.data.features.stats
Helper functions for building engineered features for team stats.

## src.data.raw.weather
Helper functions for fetching raw weather data.

## src.data.raw.games
Helper functions for fetching and saving NFL game data.

## src.data.train.target
Helper functions for building target column for games data.

## src.data.train.train
Helper functions for building the training data.

## src.docs.build
Extracts docstrings from source files and builds the documentation.

## src.model.pipeline
Helper functions for building scikit-learn pipelines.

## src.model.process
Helper functions for processing training data in a scikit pipeline.

Instead of creating bespoke classes that inherit from scikit, we'll rely on the FunctionTransformer to make them compatible with scikit pipelines. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

The preprocessing function is reserved for model-agnostic transformations that should be done at the start of a pipeline.


