from ultralytics import YOLO
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # absolute path to App folder
    data_yaml_path = os.path.join(base_dir, 'data.yaml')

    model = YOLO(os.path.join(base_dir, 'yolov9t.pt'))

    model.train(
        data=data_yaml_path,          # Use absolute path here
        epochs=100,
        imgsz=640,
#        optimizer="SGD",
        batch=8,
        mosaic=1.0,
        mixup=0.1,
        project=os.path.join(base_dir, 'runs'),  # Optional: output relative to App
        name="argus_object_detection"
    )

if __name__ == "__main__":
    main()
