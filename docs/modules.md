# Modules

## src.utils
**Path:** `src/utils.py`

Utility functions. These are all functions which are used in mutliple places throughout the source code and cannot be coupled to any particular module.

## src.__init__
**Path:** `src/__init__.py`

None

## src.data.preprocess
**Path:** `src/data/preprocess.py`

A set of functions for processing training data after it has been compiled (but before sending it to the pipeline).

## src.data.build
**Path:** `src/data/build.py`

Handles fetching raw data, building features, and assembling training data.

build.py is the main script for building the training data. It coordinates all of the paths it receives from setup.sh with helper functions for fetching/building data.

Notes
-----
Feature data creation 


## src.data.features.travel
**Path:** `src/data/features/travel.py`

None

## src.data.features.weather
**Path:** `src/data/features/weather.py`

None

## src.data.features.stats
**Path:** `src/data/features/stats.py`

None

## src.data.raw.weather
**Path:** `src/data/raw/weather.py`

Functions for fetching and processing weather data.

## src.data.raw.games
**Path:** `src/data/raw/games.py`

None

## src.data.train.target
**Path:** `src/data/train/target.py`

None

## src.data.train.train
**Path:** `src/data/train/train.py`

None

## src.docs.build
**Path:** `src/docs/build.py`

Extract docstrings from source files and build the documentation.

