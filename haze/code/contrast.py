import cv2
import os

def enhance_contrast(image_path, output_path):
    """
    Enhances the contrast of an RGB image using histogram equalization.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the contrast-enhanced image.
    """
    # Load the RGB image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Split the image into R, G, B channels
    r, g, b = cv2.split(image)
    
    # Apply histogram equalization to each channel
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    
    # Merge the equalized channels back into an RGB image
    enhanced_image = cv2.merge([r_eq, g_eq, b_eq])
    
    # Save the enhanced image
    cv2.imwrite(output_path, enhanced_image)
    print(f"Saved contrast-enhanced image to {output_path}")


if __name__ == "__main__":
    # Directory containing input RGB images
    input_dir = r"C:\Users\oishi\Documents\haze\haze\out"
    output_dir = r"C:\Users\oishi\Documents\haze\haze\contrast_enhanced_out"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):  # Adjust extensions as needed
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Enhance the contrast of the image
            enhance_contrast(input_path, output_path)