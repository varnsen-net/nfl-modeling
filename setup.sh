#!/bin/bash
set -e
export PYTHONPATH=$(pwd)

# add some style
fancy_echo() {
    local input="$1"
    local flair="─ ⋆⋅☆⋅⋆ ─"
    local border_length=$(( ${#input} - ${#flair}))
    local bar=$(printf '─%.0s' $(seq 1 $border_length))
    echo "┌${bar}${flair}┐"
    echo " $input "
    echo "└${flair}${bar}┘"
}

# base paths
SOURCE="src"
DATA="data"
RAW="$DATA/raw"
FEATURES="$DATA/features"
ANCILLARY="$DATA/ancillary"
TRAIN="$DATA/train"

# file paths
CONFIG="$SOURCE/config.json"
CITY_COORDS="$ANCILLARY/city-coordinates.csv"
RAW_GAMES="$RAW/games"
RAW_WEATHER="$RAW/weather"
RAW_TRAIN="$TRAIN/train"

# common arguments to pass to python data scripts
BUILD_ARGS="-c $CONFIG -g $RAW_GAMES.csv -w $RAW_WEATHER.csv -cc $CITY_COORDS"


# handle virtualenv
if [ ! -d "venv" ]; then
    fancy_echo "Creating virtualenv"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

if [ ! -n "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
    fancy_echo "Activated $VIRTUAL_ENV"
fi

if ! cmp -s <(sort requirements.txt) <(pip freeze | sort); then
    fancy_echo "Pip freeze does not match requirements.txt. Installing packages."
    pip install -r requirements.txt
fi


# fetch latest raw data
fancy_echo "Fetching latest raw data."
mkdir -p "$RAW"
python "$SOURCE/$RAW_GAMES.py" -g "$RAW_GAMES.csv"
echo "Raw games written to $RAW_GAMES.csv"


# update raw weather data
fancy_echo "Updating raw weather data."
python "$SOURCE/$RAW_WEATHER.py" $BUILD_ARGS
echo "Raw weather written to $RAW_WEATHER.csv"


# recurvisely build features data
fancy_echo "Building features data."
echo "Build arguments: $BUILD_ARGS"
find "$SOURCE/$FEATURES" -type f -name "*.py" | while read -r py_file; do
    relative_path=${py_file#"$SOURCE/$DATA/"}
    target_dir="$DATA/${relative_path%.py}"
    echo "Running $py_file and writing to $target_dir"
    mkdir -p "$target_dir"
    python "$py_file" $BUILD_ARGS -o "$target_dir"
done


# build training data
fancy_echo "Building training data."
mkdir -p "$TRAIN"
python "$SOURCE/$RAW_TRAIN.py" -c "$CONFIG" -g "$RAW_GAMES.csv" -f "$FEATURES" -o "$RAW_TRAIN.csv"
echo "Raw training data written to $RAW_TRAIN.csv"
