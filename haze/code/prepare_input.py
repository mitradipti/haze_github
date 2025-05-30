import torch
import torchvision.transforms as transforms
from PIL import Image
from blockshrm import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock

def prepare_combined_input(transmission_map_path, contrast_image_path, original_image_path):
    """
    Concatenates expanded 3-channel transmission maps, contrast-enhanced images,
    and original input images to produce a single 9-channel input tensor.

    Args:
        transmission_map_path (str): Path to the expanded 3-channel transmission map image.
        contrast_image_path (str): Path to the 3-channel contrast-enhanced image.
        original_image_path (str): Path to the original 3-channel input image.

    Returns:
        torch.Tensor: A 9-channel input tensor of shape (1, 9, height, width).
    """
    # Define a transform to convert images to tensors
    transform = transforms.ToTensor()

    # Load and transform the expanded 3-channel transmission map
    transmission_map = Image.open(transmission_map_path).convert("RGB")  # Ensure it's 3 channels
    transmission_map_tensor = transform(transmission_map)
    print("Transmission Map Shape (before resizing):", transmission_map_tensor.shape)

    # Load and transform the 3-channel contrast-enhanced image
    contrast_image = Image.open(contrast_image_path).convert("RGB")
    contrast_image_tensor = transform(contrast_image)
    print("Contrast Image Shape (before resizing):", contrast_image_tensor.shape)
    
    # Load and transform the original 3-channel input image
    original_image = Image.open(original_image_path).convert("RGB")
    original_image_tensor = transform(original_image)
    print("Original Image Shape (before resizing):", original_image_tensor.shape)
    
    # Ensure all tensors have the same spatial dimensions
    height, width = transmission_map_tensor.shape[1:]
    contrast_image_tensor = transforms.Resize((height, width))(contrast_image_tensor)
    original_image_tensor = transforms.Resize((height, width))(original_image_tensor)

    print("Contrast Image Shape (after resizing):", contrast_image_tensor.shape)
    print("Original Image Shape (after resizing):", original_image_tensor.shape)

    # Concatenate along the channel dimension
    combined_input = torch.cat([transmission_map_tensor, contrast_image_tensor, original_image_tensor], dim=0)

    # Add a batch dimension (batch size = 1)
    return combined_input.unsqueeze(0)

# Paths to your images
transmission_map_path = r"C:\Users\oishi\Documents\haze\haze\tmp_outr_predicted\01_outdoor_hazy.jpg"
contrast_image_path = r"C:\Users\oishi\Documents\haze\haze\contrast_enhanced_out\01_outdoor_hazy.jpg"
original_image_path = r"C:\Users\oishi\Documents\haze\haze\out\01_outdoor_hazy.jpg"
# Prepare the combined input tensor
combined_input = prepare_combined_input(transmission_map_path, contrast_image_path, original_image_path)

print("Combined Input Shape:", combined_input.shape)
