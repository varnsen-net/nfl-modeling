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
    if [ ! -d "$path" ]; then
        mkdir -p $path
        echo "Created $path"
    fi
done


# build data
export PYTHONPATH=`pwd`
echo "Fetching raw data..."
python src/data/fetch_raw_data.py
echo "Building features..."
python src/features/build_features.py


