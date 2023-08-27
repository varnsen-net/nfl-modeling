#!/bin/bash


# handle virtualenv
if [ ! -d "venv" ]; then
    echo "Creating virtualenv"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

if [ ! -n "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
    echo "Activated $VIRTUAL_ENV"
fi

if ! cmp -s <(sort requirements.txt) <(pip freeze | sort); then
    echo "╭───────────────────────────────────────────────────────────.★..─╮"
    echo " Pip freeze does not match requirements.txt. Installing packages."
    echo "╰─..★.───────────────────────────────────────────────────────────╯"
    pip install -r requirements.txt
fi


# create data dirs if they do not exist
for path in $(jq -r '.paths[]' config.json); do
    dir=$(dirname $path)
    if [ ! -d "$dir" ]; then
        mkdir -p $dir
        echo "Created $dir"
    fi
done


# build data
export PYTHONPATH=`pwd`
echo "Fetching raw data..."
python src/data/raw/fetch_raw_data.py
echo "Building features..."
python src/data/features/build_features.py
echo "Building training data..."
python src/data/training/build_training_data.py


