.PHONY: data

data:
	mkdir -p data/raw/games
	mkdir -p data/features/stats
	mkdir -p data/features/travel
	mkdir -p data/features/weather
	python src/data/fetch_raw_data.py
	python src/features/build_features.py
