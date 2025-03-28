import os
import cv2
import numpy as np
from tqdm import tqdm  # For progress bar

def generate_segmented_images(image_dir, label_dir, output_dir, mask_output_dir):
    """
    Generate and save segmented images along with their masks.
    
    Parameters:
    - image_dir: Directory containing input images.
    - label_dir: Directory containing `.txt` label files.
    - output_dir: Directory to save generated segmented images.
    - mask_output_dir: Directory to save generated mask images.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing Images"):
        try:
            base_name = os.path.splitext(image_file)[0]
            label_file = os.path.join(label_dir, f"{base_name}.txt")
            image_path = os.path.join(image_dir, image_file)

            if not os.path.exists(label_file):
                print(f"Warning: No label file for {image_file}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot load image {image_file}")
                continue
            h, w, _ = image.shape

            mask_image = np.zeros((h, w, 3), dtype=np.uint8)  # Black background

            with open(label_file, "r") as file:
                for line in file:
                    try:
                        data = list(map(float, line.strip().split()))
                        if len(data) != 5:
                            raise ValueError(f"Invalid label format: {data}")

                        _, x_center, y_center, bbox_width, bbox_height = data

                        # Convert YOLO format to pixel values
                        x_center, y_center = int(x_center * w), int(y_center * h)
                        bbox_width, bbox_height = int(bbox_width * w), int(bbox_height * h)

                        # Get bounding box coordinates
                        x1, y1 = x_center - bbox_width // 2, y_center - bbox_height // 2
                        x2, y2 = x_center + bbox_width // 2, y_center + bbox_height // 2

                        # Ensure coordinates are within image boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # Draw rectangle on the mask
                        cv2.rectangle(mask_image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

                    except ValueError as e:
                        print(f"Skipping label in {label_file} due to error: {e}")
                        continue

            mask_output_path = os.path.join(mask_output_dir, f"{base_name}.jpg")
            cv2.imwrite(mask_output_path, mask_image)

            # Apply mask to original image
            segmented_image = cv2.bitwise_and(image, mask_image)

            segmented_output_path = os.path.join(output_dir, f"{base_name}_segmented.jpg")
            cv2.imwrite(segmented_output_path, segmented_image)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Example usage
image_dir = "Dataset/bean/ALS/images/"
label_dir = "Dataset/bean/ALS/labels/"
output_dir = "Dataset/bean/ALS/segmented/"
mask_output_dir = "Dataset/bean/ALS/masks/"

generate_segmented_images(image_dir, label_dir, output_dir, mask_output_dir)