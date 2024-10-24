# SWIFT (Sample-Weighted Inductive Football Transformer)

![SWIFT logo](https://i.imgur.com/IgZi9pt.png)

Welcome to the repo for SWIFT, a machine learning model designed to produce well-calibrated probabilities for NFL games.

Current status: trains a few models and tunes hyperparameters with the hyperopt library.

## Getting started
Simply `bash run.sh` and you're off to the races!

After training, you can view model scores in `/data/results`

If you want to generate predictions on upcoming games, specify the current NFL week and season in `/src/config/config.py`

## Documentation

For source code documentation, see `docs/build/html/index.html`

## FAQ
**Q: Is your model any good?**

A: Probably not.

**Q: Your acronym doesn't make any sense.**

A: That's not a question.

**Q: How come you don't use efficiency metrics like EPA/WPA?**

A: Data leakage! These metrics are derived from all currently available NFL data and would give the model an unfair glimpse of the future.

## Current engineered features
- home/away rest
- travel distances
- pythagorean expectation
- points per game
- points per drive
- penalty yards per drive
- series success rate
- net yards per play 

Feature engineering makes adjustments for league averages and opponent strength. Most features are expressed in terms of median absolute deviations from the adjusted team means for that stat.

## Training procedure
SWIFT does everything possible to avoid data leakage. It should never get a glimpse into the future.

#### Pipeline:
1. Transform the home/away team structure into an object/adversary team structure so that the model tries to predict a 50/50 mix of home and away win probabilities.
2. Train and evaluate using a grouped time series cross-validation scheme. The model trains on a block of *m* consecutive seasons, then validates on the following *n* seasons.
3. On each training fold in the time series cv, calibrate model probabilities with a 5-fold cv.
4. Search for optimal hyperparameters with hyperopt and minimize the average brier score.
5. Train final model on full dataset using optimal hyperparameters.
6. Evaluate on holdout data.

## Future tasks
- ~~Auto-generate API docs with Sphinx~~
- ~~Create engineered features for drive efficiency~~
- Complete documentation for all modules and functions
- Account for quarterback injuries
- Tidy up some bits of rushed code
- Complete unit tests
- Expand model evaluation to include tracking optimal hyperparameters
- Write bespoke time series cross validation windows that look forward **and** backward
- Add a mixed-effects logreg model that controls for season
- Add a regression model that predicts game score differentials
