# Project Overview

### This project is called AI Powered Cheating Detector project that uses Yolo Object Detection in detecting various objects like, phone, calculator, smartwatch, watch, and chairs. This project also uses Yolo Pose Estimation in detecting cheating behaviors during an exams.

### To train the AI, go to "machine_learning" folder and to check the working app after training simply go to "system" folder.

# Install requirements on both folders

### For machine_learning

cd machine_learning, then:<br>
pip install -r requirements.txt

### For system

cd system, then:<br>
pip install -r requirements.txt

# This app is using flask, here's how to run:

flask --app wcapp.py run

### or

flask --app wcapp run --host 0.0.0.0 --port=5000

# When updating the project every time there is an update

git fetch origin main<br>
git checkout .

### For individual folders like machine_learning/system

git checkout origin/main -- ./machine_learning/<br>
git checkout origin/main -- ./system/
