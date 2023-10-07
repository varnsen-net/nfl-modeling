# SWIFT (Sample-Weighted Inductive Football Transformer)

![SWIFT logo](https://i.imgur.com/IgZi9pt.png)

Welcome to the repo for SWIFT, a machine learning model to predict NFL outcomes.

## Description
SWIFT is a model designed to output well-calibrated probabilities for NFL games. It's still in the very early stages of development. Many features are still missing.

Current status: trains a LightGBM classifier and tunes hyperparameters with the hyperopt library.

## Getting started
Simply `bash setup.sh` and you're off to the races!

After training, you can view model scores in `/data/results`

## Current engineered features
- home/away rest
- field stats
- weather
- away travel
- pythagorean expectation
- elo rating
- pass/rush efficiency metrics

## FAQ

**Q: Is your model any good?**

A: Probably not.

**Q: Your acronym doesn't make any sense.**

A: That's not a question.
