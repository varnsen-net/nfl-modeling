# SWIFT (Sample-Weighted Inductive Football Transformer)

![SWIFT logo](https://i.imgur.com/IgZi9pt.png)

Welcome to the repo for SWIFT, a machine learning model to predict NFL outcomes.

## Description
SWIFT is a model designed to output well-calibrated probabilities for NFL games. It's still in the very early stages of development, so many things are still missing.

Current status: can train a baseline and advanced model and record various scoring metrics for each run.

## Getting started
Simply `bash setup.sh` and you're off to the races!

After training, you can view model scores in `./data/results`

## Current engineered features
- home/away rest
- field stats
- weather
- away travel
- pythagorean expectation
- elo rating

## FAQ

**Q: Is your model any good?**

A: Probably not.

**Q: Your acronym doesn't make any sense.**

A: That's not a question.
