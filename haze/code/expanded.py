from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import ImageEnhance
import matplotlib.pyplot as plt


class HazyImageDataset(Dataset):
    def __init__(self, hazy_image_dir, transmission_map_dir, transform=None):
        self.hazy_image_dir = hazy_image_dir
        self.transmission_map_dir = transmission_map_dir
        self.transform = transform
        self.hazy_image_filenames = os.listdir(hazy_image_dir)

    def __len__(self):
        return len(self.hazy_image_filenames)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_image_dir, self.hazy_image_filenames[idx])
        transmission_map_path = os.path.join(self.transmission_map_dir, self.hazy_image_filenames[idx])

        # Load hazy image and transmission map
        hazy_image = Image.open(hazy_image_path).convert('RGB')
        transmission_map = Image.open(transmission_map_path).convert('L')

        # Generate contrast-enhanced image using histogram equalization
        contrast_enhanced_image = self.apply_histogram_equalization(hazy_image)

        # Apply transformations
        if self.transform:
            hazy_image = self.transform(hazy_image)
            contrast_enhanced_image = self.transform(contrast_enhanced_image)
            transmission_map = self.transform(transmission_map)

        # Expand transmission map to three channels
        if transmission_map.dim() == 2:  # If shape is [H, W], add a channel dimension
            transmission_map = transmission_map.unsqueeze(0)  # Shape: [1, H, W]
        transmission_map = transmission_map.repeat(3, 1, 1)  # Shape: [3, H, W]

        # Concatenate hazy image, contrast-enhanced image, and transmission map
        nine_channel_input = torch.cat([hazy_image, contrast_enhanced_image, transmission_map], dim=0)

        # Debugging output
        print(f"Nine-Channel Input Shape: {nine_channel_input.shape}")

        return nine_channel_input, hazy_image, contrast_enhanced_image  # Return contrast-enhanced image as well

    @staticmethod
    def apply_histogram_equalization(hazy_image):
        # Convert PIL image to NumPy array
        hazy_image_np = np.array(hazy_image)

        # Apply histogram equalization to each channel
        if len(hazy_image_np.shape) == 3:  # RGB image
            r, g, b = cv2.split(hazy_image_np)
            r_eq = cv2.equalizeHist(r)
            g_eq = cv2.equalizeHist(g)
            b_eq = cv2.equalizeHist(b)
            contrast_enhanced_image_np = cv2.merge((r_eq, g_eq, b_eq))
        else:  # Grayscale image
            contrast_enhanced_image_np = cv2.equalizeHist(hazy_image_np)

        # Convert NumPy array back to PIL image
        contrast_enhanced_image = Image.fromarray(contrast_enhanced_image_np)
        enhancer = ImageEnhance.Contrast(contrast_enhanced_image)
        contrast_enhanced_image = enhancer.enhance(2.0)
        return contrast_enhanced_image


# Save all contrast-enhanced images
if __name__ == "__main__":
    # Define directories
    hazy_image_dir = r'C:\Users\oishi\Documents\haze\haze\resized_input_images\resized_input_images'
    transmission_map_dir = r'C:\Users\oishi\Documents\haze\haze\transmission_maps\transmission_maps'
    output_dir = r'C:\Users\oishi\Documents\haze\haze\contrast_enhanced_images'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Instantiate the dataset
    dataset = HazyImageDataset(hazy_image_dir, transmission_map_dir, transform=transform)


    # Print dataset size
    print(f"Dataset size: {len(dataset)}")

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Use batch_size=1 for visualization

    # Iterate through the DataLoader
    for nine_channel_input, hazy_image, contrast_enhanced_image in dataloader:
        # Convert tensors to NumPy arrays for visualization
        hazy_image_np = hazy_image.squeeze(0).permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        contrast_enhanced_image_np = contrast_enhanced_image.squeeze(0).permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]

        # Save contrast-enhanced images
    for idx in range(len(dataset)):
        _, _, contrast_enhanced_image = dataset[idx]

        # Convert the contrast-enhanced image back to a PIL image
        contrast_enhanced_image_pil = transforms.ToPILImage()(contrast_enhanced_image)

        # Save the contrast-enhanced image
        output_path = os.path.join(output_dir, f"contrast_enhanced_{idx + 1}.jpg")
        contrast_enhanced_image_pil.save(output_path)
        print(f"Saved contrast-enhanced image: {output_path}")