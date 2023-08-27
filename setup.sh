# set PYTHONPATH to current directory
export PYTHONPATH=`pwd`


# check to see if user has already created a python virtual environment.
# if not, create one and install the required packages.
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi


# alert user if they have not activated the virtual environment
if [ ! -n "$VIRTUAL_ENV" ]; then
    echo "Please activate your virtual environment."
    echo "Run the following command: source venv/bin/activate"
    exit 1
fi


