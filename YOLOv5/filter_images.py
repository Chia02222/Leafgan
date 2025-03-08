import os
import shutil

# Define paths
image_folder = "Dataset/strawberry/leafspot/images"  # Folder containing all images
mask_folder = "Dataset/strawberry/leafspot/masks"      # Folder containing mask images
output_folder = "Dataset/strawberry/leafspot/filtered_images"  # Folder to save found mask images

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get list of original image names from mask filenames (without extensions)
mask_files = {os.path.splitext(f)[0].replace("_mask", "") for f in os.listdir(mask_folder)}

# Filter and copy only images that have corresponding masks
for image_file in os.listdir(image_folder):
    image_name, ext = os.path.splitext(image_file)
    
    # Check if any variation of the image name exists in the mask list
    if any(image_name == mask_name for mask_name in mask_files):
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        
        # Copy image to the new folder
        shutil.copy(image_path, output_path)
        print(f"Copied: {image_file}")

print("Filtered images saved successfully.")
