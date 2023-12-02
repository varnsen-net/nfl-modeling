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

# paths
SOURCE="src"
MODELING="$SOURCE/model"


# handle virtualenv
if [ ! -d "venv" ]; then
    fancy_echo "Creating virtualenv"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install wheel
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


# build data
echo -n "Build data? [y/N]: "
read answer
if [[ $answer =~ ^[Yy]$ ]]; then
    fancy_echo "Building data"
    python3 $SOURCE/data/build.py
fi


# train models
echo -n "Train models? [y/N]: "
read answer
if [[ $answer =~ ^[Yy]$ ]]; then
    fancy_echo "Training models"
    python3 $MODELING/train.py
fi
