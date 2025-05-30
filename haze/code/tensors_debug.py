import torch
from torchvision import transforms
from PIL import Image

# Debbugging and testing
# Load the expanded 3-channel image
image_path = r"C:\Users\oishi\Documents\haze\haze\tmp_outr_threecha\01_outdoor_hazy.jpg"
image = Image.open(image_path)

# Convert to PyTorch tensor
transform = transforms.ToTensor()
tensor_image = transform(image)

print("Tensor Shape:", tensor_image.shape)