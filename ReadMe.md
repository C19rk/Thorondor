# Install requirements on both folders

### For machine_learning

cd machine_learning, then:
python -m pip install -r requirements.txt

### For system

cd system, then:
python -m pip install -r requirements.txt

# This app is using flask, here's how to run:

flask --app wcapp.py run

### or

flask --app wcapp run --host 0.0.0.0 --port=5000

# When updating the project every time there is an update

git fetch origin main
git checkout .

### For individual folders like machine_learning

git checkout origin/main -- ./machine_learning/

### For system

git checkout origin/main -- ./system/
