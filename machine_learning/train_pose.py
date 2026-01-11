from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 pose model
    model = YOLO("yolov9t-pose.pt")  # change to s/m/l if you have more GPU

    # Train
    model.train(
        data="datapose.yaml",   # dataset YAML
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,             # GPU 0; set to "cpu" if no GPU
        project="runs/pose",  # where to save runs
        name="aidestimation"
    )

if __name__ == "__main__":
    main()
