# SWIFT (Sample-Weighted Inductive Football Transformer)

![SWIFT logo](https://i.imgur.com/IgZi9pt.png)

Welcome to the repo for SWIFT, a machine learning model designed to produce well-calibrated probabilities for NFL games.

Current status: trains a LightGBM classifier and tunes hyperparameters with the hyperopt library.

Gonna take some time off from active development in this repo for a few months. Good luck with your playoff bets!

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

**Q: How come you don't use efficiency metrics like pythagorean expectation/EPA/WPA?**

A: Data leakage! These metrics are derived from all currently available NFL data and would give the model an unfair glimpse of the future.

## Current engineered features
- home/away rest
- field stats
- travel distances
- opponent-adjusted points for/against
- basic pass/rush statistics

## Training procedure
SWIFT does everything possible to avoid data leakage. It should never get a glimpse into the future.

#### Pipeline:
1. Transform the home/away team structure into an object/adversary team structure so that the model tries to predict a 50/50 mix of home and away win probabilities.
2. Train and evaluate using a grouped time series cross-validation scheme. The model trains on a block of *m* consecutive seasons, then validates on the following *n* seasons.
3. On each training fold in the time series cv, calibrate model probabilities with a 5-fold cv.
4. Search for optimal hyperparameters with hyperopt and minimize the average brier score.
5. Evaluate on holdout data using the ensemble of cross-validated estimators from step 2.
6. Train final model on full dataset.

## Future tasks
- ~~Auto-generate API docs with Sphinx~~
- ~~Complete documentation for all modules and functions~~
- ~~Engineer 'normal' gametime features, i.e. when score and remaining time do not affect play-calling~~
- Tidy up some bits of rushed code
- Complete unit tests
- Expand model evaluation to include tracking optimal hyperparameters
- Write bespoke time series cross validation windows that look forward **and** backward
- Account for quarterback injuries
- Create engineered features for drive efficiency
- Add a mixed-effects logreg model that controls for season
