# Modules

## src.utils
**Path:** `src/utils.py`

Utility functions. These functions are used in mutliple places throughout the source code and cannot be coupled to any particular module.

## src.__init__
**Path:** `src/__init__.py`

None

## src.data.build
**Path:** `src/data/build.py`

Builds the full set of training data from scratch.

Receives a set of paths from setup.sh, then fetches raw data, builds features, and assembles training data.


## src.data.features.travel
**Path:** `src/data/features/travel.py`

Helper functions for building engineered features for team travel.

## src.data.features.weather
**Path:** `src/data/features/weather.py`

Helper functions for building weather features.

## src.data.features.stats
**Path:** `src/data/features/stats.py`

Helper functions for building engineered features for team stats.

## src.data.raw.weather
**Path:** `src/data/raw/weather.py`

Helper functions for fetching raw weather data.

## src.data.raw.games
**Path:** `src/data/raw/games.py`

Helper functions for fetching and saving NFL game data.

## src.data.train.target
**Path:** `src/data/train/target.py`

Helper functions for building target column for games data.

## src.data.train.train
**Path:** `src/data/train/train.py`

Helper functions for building the training data.

## src.docs.build
**Path:** `src/docs/build.py`

Extracts docstrings from source files and builds the documentation.

## src.model.process
**Path:** `src/model/process.py`

Helper functions for processing training data in a scikit pipeline.

Instead of creating bespoke classes that inherit from scikit, we'll rely on the FunctionTransformer to make them compatible with scikit pipelines.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html


