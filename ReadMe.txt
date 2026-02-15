__________________________Project Overview__________________________

This project is called AI Powered Cheating Detector project that uses Yolo Object Detection in
detecting various objects like, phone, calculator, smartwatch, watch, and chairs. This project
also uses Yolo Pose Estimation in detecting cheating behaviors during an exams.

To train the AI, go to "machine_learning" folder and to check the working app after training
simply go to "system" folder.



__________________________Caution!!__________________________
When training data, you might want to watch out for the data's label IDs. To update the IDs,
simply run change.py in "change id". Ensure that test, train, and valid folder's label's IDs
are fully updated not just one folder.



__________________________Install requirements on both folders__________________________

============= For machine_learning =============

cd machine_learning

then:
pip install -r requirements.txt

============= For system =============

cd system

then:
pip install -r requirements.txt



__________________________This app is using flask, here's how to run:__________________________

============= For webcam =============

flask --app wcapp.py run

or

flask --app wcapp run --host 0.0.0.0 --port=5000

============= For an actual camera device =============

flask --app app.py run

or

flask --app app run --host 0.0.0.0 --port=5000



__________________________When updating the project every time there is an update__________________________

git fetch origin main
git checkout .


============= For individual folders like machine_learning/system =============

git checkout origin/main -- ./machine_learning/
git checkout origin/main -- ./system/
