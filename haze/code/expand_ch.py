import os
import cv2
import numpy as np

def expand_to_three_channels(image_path, output_path):
    """
    Expands a 1-channel grayscale image to a 3-channel image and saves it.

    Args:
        image_path (str): Path to the input 1-channel grayscale image.
        output_path (str): Path to save the 3-channel expanded image.
    """
    # Load the grayscale image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Expand to 3 channels by stacking the grayscale image
    three_channel_image = cv2.merge([grayscale_image, grayscale_image, grayscale_image])
    
    # Save the 3-channel image
    cv2.imwrite(output_path, three_channel_image)
    print(f"Saved 3-channel image to {output_path}")


if __name__ == "__main__":
    # Directory containing 1-channel grayscale transmission maps
    input_dir = r"C:\Users\oishi\Documents\haze\haze\tmp_outr_predicted"
    output_dir = r"C:\Users\oishi\Documents\haze\haze\tmp_outr_threecha"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each grayscale image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):  # Adjust extensions as needed
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Expand the grayscale image to 3 channels
            expand_to_three_channels(input_path, output_path)

