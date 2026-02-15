from ultralytics import YOLO
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    yaml_path = os.path.join(base_dir, "data_pose.yaml")

    model = YOLO("yolov8n-pose.pt")

    model.train(
        data=yaml_path,        
        epochs=100,            
        imgsz=640,             
        batch=-1,              
        device=0,              # GPU 0; set to "cpu" if no GPU
        patience=20,           
        project="runs/pose",   
        name="argus_pose_estimation", 
        exist_ok=True,         
        plots=True             
    )

if __name__ == "__main__":
    main()