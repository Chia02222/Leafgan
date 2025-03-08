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
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Iterate over all image files in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in tqdm(image_files, desc="Processing Images"):
        try:
            # Define corresponding label file path
            base_name = os.path.splitext(image_file)[0]
            label_file = os.path.join(label_dir, f"{base_name}.txt")
            image_path = os.path.join(image_dir, image_file)

            # Check if the label file exists
            if not os.path.exists(label_file):
                print(f"Warning: No corresponding label file for {image_file}")
                continue

            # Load the original image to get its dimensions
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot load image {image_file}")
                continue
            h, w, _ = image.shape

            # Create a black background image for the mask
            mask_image = np.zeros((h, w, 3), dtype=np.uint8)  # Black background

            # Read the `.txt` label file
            with open(label_file, "r") as file:
                for idx, line in enumerate(file):
                    try:
                        # Parse the line into a list of floats
                        data = list(map(float, line.strip().split()))
                        if len(data) < 3:  # At least class_id, one (x, y) pair, and confidence
                            raise ValueError(f"Line {idx + 1} in {label_file} is too short: {data}")

                        points = np.array(data[1:-1], dtype=np.float32)  # Exclude class_id and confidence

                        # Ensure points can be reshaped into (x, y) pairs
                        if len(points) % 2 != 0:
                            print(f"Warning: Line {idx + 1} in {label_file} has an odd number of coordinates. Ignoring the last value.")
                            points = points[:-1]  # Drop the last value to make it even

                        if len(points) < 4:  # Must have at least one (x, y) pair
                            raise ValueError(f"Line {idx + 1} in {label_file} has too few points after cleanup: {points}")

                        points = points.reshape(-1, 2)
                        points = np.round(points * [w, h]).astype(int)  # Scale to image dimensions

                        # Draw the polygon on the mask image
                        cv2.fillPoly(mask_image, [points], color=(255, 255, 255))  # Mask color is white

                    except ValueError as e:
                        print(f"Skipping line {idx + 1} in {label_file} due to error: {e}")
                        continue

            # Save the mask image separately
            mask_output_path = os.path.join(mask_output_dir, f"{base_name}_mask.jpg")
            cv2.imwrite(mask_output_path, mask_image)

            # Apply the mask to the original image
            segmented_image = cv2.bitwise_and(image, mask_image)

            # Save the generated segmented image
            segmented_output_path = os.path.join(output_dir, f"{base_name}_segmented.jpg")
            cv2.imwrite(segmented_output_path, segmented_image)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Example usage
image_dir = "Dataset/coffee/rust/images/"  # Directory containing input images
label_dir = "Dataset/coffee/rust/labels/"  # Directory containing label files
output_dir = "Dataset/coffee/rust/segmented/"  # Directory to save segmented images
mask_output_dir = "Dataset/coffee/rust/masks/"  # Directory to save mask images

generate_segmented_images(image_dir, label_dir, output_dir, mask_output_dir)
