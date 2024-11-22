#!/bin/bash
set -e


BUILD="./docs/build/"
SOURCE_API="./docs/source/api/"
VENV_DIR="venv"

# add some fresh style to echo
fancy_echo() {
    local input="$1"
    local flair="─ ⋆⋅☆⋅⋆ ─"
    local border_length=$(( ${#input} - ${#flair}))
    local bar=$(printf '─%.0s' $(seq 1 $border_length))
    echo "┌${bar}${flair}┐"
    echo " $input "
    echo "└${flair}${bar}┘"
}

# activate virtual environment
if [ ! -n "$VIRTUAL_ENV" ]; then
    fancy_echo "Activating $VIRTUAL_ENV"
    source $VENV_DIR/bin/activate
fi

# remove directories
fancy_echo "Removing old build and source files..."
rm -drf $BUILD
rm -drf $SOURCE_API

# build documentation
fancy_echo "Running make html..."
if command -v make &> /dev/null; then
    if [ -d ./docs/ ]; then
	cd docs
	make html
	cd ..
    else
	echo "./docs/ directory not found."
    fi
else
    echo "Make command not found."
fi

