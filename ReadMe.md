# Install requirements on both folders

### For machine_learning

cd machine_learning, then:
pip install -r requirements.txt

### For system

cd system, then:
pip install -r requirements.txt

# This app is using flask, here's how to run:

flask --app wcapp.py run

### or

flask --app wcapp run --host 0.0.0.0 --port=5000
