import os
import cv2

def resize_images(input_dir, output_dir, size=(1600, 1200)):
    """
    Resizes all images in the input directory to the specified size and saves them to the output directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory to save resized images.
        size (tuple): Desired size for the resized images (width, height).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # Adjust extensions as needed
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Could not load image: {input_path}")
                continue

            # Resize the image
            resized_image = cv2.resize(image, size)

            # Save the resized image
            cv2.imwrite(output_path, resized_image)
            print(f"Resized and saved image to {output_path}")

if __name__ == "__main__":
    # Directory containing input images
    input_dir = r"C:\Users\oishi\Documents\haze\haze\o_haze\o_haze\GT"
    # Directory to save resized images
    output_dir = r"C:\Users\oishi\Documents\haze\haze\gt"

    # Resize images to 256x256
    resize_images(input_dir, output_dir, size=(1600, 1200))