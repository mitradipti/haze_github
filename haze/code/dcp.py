import cv2
import numpy as np
import os

def dark_channel_prior(image, size=15):
    """
    Computes the dark channel prior of an image.

    Args:
        image (numpy.ndarray): Input image (normalized to [0, 1]).
        size (int): Size of the structuring element for erosion.

    Returns:
        numpy.ndarray: Dark channel of the image.
    """
    min_channel = np.amin(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def atmospheric_light(image, dark_channel):
    """
    Estimates the atmospheric light in the image.

    Args:
        image (numpy.ndarray): Input image (normalized to [0, 1]).
        dark_channel (numpy.ndarray): Dark channel of the image.

    Returns:
        numpy.ndarray: Estimated atmospheric light.
    """
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    search_idx = (-flat_dark).argsort()[:int(0.001 * len(flat_dark))]
    atmospheric_light = np.mean(flat_image[search_idx], axis=0)
    return atmospheric_light

def transmission_map(image, atmospheric_light, omega=0.95, size=15):
    """
    Computes the transmission map of the image.

    Args:
        image (numpy.ndarray): Input image (normalized to [0, 1]).
        atmospheric_light (numpy.ndarray): Estimated atmospheric light.
        omega (float): Scattering coefficient (default: 0.95).
        size (int): Size of the structuring element for erosion.

    Returns:
        numpy.ndarray: Transmission map of the image.
    """
    norm_image = image / atmospheric_light
    transmission = 1 - omega * dark_channel_prior(norm_image, size)
    return transmission

def refine_transmission(image, transmission, radius=60, eps=1e-3):
    """
    Refines the transmission map using a guided filter.

    Args:
        image (numpy.ndarray): Input image (normalized to [0, 1]).
        transmission (numpy.ndarray): Initial transmission map.
        radius (int): Radius of the guided filter.
        eps (float): Regularization parameter for the guided filter.

    Returns:
        numpy.ndarray: Refined transmission map.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    guided_filter = cv2.ximgproc.createGuidedFilter(gray, radius, eps)
    refined_transmission = guided_filter.filter(transmission)
    return refined_transmission

if __name__ == "__main__":
    # Directory containing input hazy images
    input_dir = r'C:\Users\oishi\Documents\haze\haze\out'
    output_dir = r'C:\Users\oishi\Documents\haze\haze\tmp_out'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Log all files in the input directory
    print("Files in input directory:")
    for file in os.listdir(input_dir):
        print(file)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):  # Only process .jpg files
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                print(f"Processing: {filename}")

                # Load image
                image = cv2.imread(input_path) / 255.0
                if image is None:
                    print(f"Error: Could not load image {input_path}")
                    continue

                image = image.astype(np.float32)

                # Estimate dark channel
                dark_channel = dark_channel_prior(image)

                # Estimate atmospheric light
                A = atmospheric_light(image, dark_channel)

                # Estimate transmission map
                transmission = transmission_map(image, A)

                # Refine transmission map using guided filter
                refined_transmission = refine_transmission(image, transmission)

                # Save the refined transmission map
                refined_transmission_uint8 = (refined_transmission * 255).astype(np.uint8)
                cv2.imwrite(output_path, refined_transmission_uint8)
                print(f"Saved transmission map to {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")