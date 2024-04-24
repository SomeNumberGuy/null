import cv2
import glob
import os

def apply_binary_threshold(image):
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    return thresholded

def process_images(folder_path):
    # Create output folder if it doesn't exist
    output_folder = folder_path #os.path.join(folder_path, "thresholded_images")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all image file paths in the folder
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))

    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply binary thresholding
        thresholded_image = apply_binary_threshold(image)

        # Get filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Save thresholded image with suffix
        output_path = os.path.join(output_folder, filename + ".png")
        cv2.imwrite(output_path, thresholded_image)

        print(f"Saved {output_path}")

if __name__ == "__main__":
    folder_path = "../auto-seg/mask"
    process_images(folder_path)
