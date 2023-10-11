RAW_DATA_URLS = {
    "games": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "elos": "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv",
    "weather_hist": "https://archive-api.open-meteo.com/v1/archive",
    "weather_forecast": "https://api.open-meteo.com/v1/forecast",
    "plays": "https://github.com/nflverse/nflverse-data/releases/download/pbp",
}

CURRENT_SEASON = 2023
CURRENT_WEEK = 5

TRAINING = {
    "holdout_year": 2020,
    "games_cols": ["game_id", "season", "week", "away_rest", "home_rest", "div_game", "roof", "surface"]
}

WEATHER_METADATA = {
    "temperature": {
        "api_name": "temperature_2m",
        "agg_func": "mean",
        "default": 21.0
    },
    "apparent_temperature": {
        "api_name": "apparent_temperature",
        "agg_func": "mean",
        "default": 21.0
    },
    "humidity": {
        "api_name": "relativehumidity_2m",
        "agg_func": "mean",
        "default": 50.0
    },
    "dewpoint": {
        "api_name": "dewpoint_2m",
        "agg_func": "mean",
        "default": 12.0
    },
    "pressure": {
        "api_name": "surface_pressure",
        "agg_func": "mean",
        "default": None
    },
    "wind_speed": {
        "api_name": "windspeed_10m",
        "agg_func": "mean",
        "default": 0.0
    },
    "cloud_cover": {
        "api_name": "cloudcover",
        "agg_func": "mean",
        "default": 100.0
    },
    "rain": {
        "api_name": "rain",
        "agg_func": "sum",
        "default": 0.0
    },
    "snow": {
        "api_name": "snowfall",
        "agg_func": "sum",
        "default": 0.0
    },
}

FEATURE_PRECISIONS = {
    "temperature_2m": 0,
    "apparent_temperature": 0,
    "relativehumidity_2m": 1,
    "dewpoint_2m": 1,
    "surface_pressure": 0,
    "windspeed_10m": 1,
    "cloudcover": 0,
    "rain": 1,
    "snowfall": 1,
    "away_lon_delta": 2,
    "away_travel_distance": 0,
    "elo": 0,
    "pythagorean_expectation": 3,
}
