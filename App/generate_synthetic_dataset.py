import os
import cv2
import random
import numpy as np

# CONFIG
output_dir = "C:/Users/Administrator/Desktop/VSCode/Research Project/Thorondor/data"
classes = ["normal", "looking_away", "using_phone"]
num_images = 20
image_size = (640, 480)

# Ensure folders exist
for split in ['train', 'val']:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

def generate_image(index, split):
    # Generate a blank white image
    img = 255 * np.ones((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Random class selection
    class_id = random.randint(0, len(classes) - 1)

    # Random coordinates and size for bounding box
    x = random.randint(100, image_size[0] - 200)
    y = random.randint(100, image_size[1] - 200)
    w = random.randint(50, 150)
    h = random.randint(100, 200)

    # Draw rectangle (bounding box)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # Save the image
    image_filename = f"img{index:03d}.jpg"
    image_path = os.path.join(output_dir, "images", split, image_filename)
    cv2.imwrite(image_path, img)

    # Check if the image was saved correctly
    if os.path.exists(image_path):  # Check if the image file exists
        print(f"Image saved: {image_path}")
    else:
        print(f"Failed to save image: {image_path}")

    # Convert bounding box to YOLO format (normalized)
    x_center = (x + w / 2) / image_size[0]
    y_center = (y + h / 2) / image_size[1]
    w_norm = w / image_size[0]
    h_norm = h / image_size[1]

    # Save the corresponding label file
    label_path = os.path.join(output_dir, "labels", split, f"img{index:03d}.txt")
    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Print the label and image info
    print(f"Generated {image_path} with class {classes[class_id]}")

# Generate images and labels
for i in range(num_images):
    split = "train" if i < int(num_images * 0.8) else "val"
    generate_image(i, split)

print("\n✅ Synthetic dataset created!")
