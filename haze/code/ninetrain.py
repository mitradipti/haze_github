from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import ImageEnhance
from blocks import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
from pytorch_msssim import ssim  # Install with `pip install pytorch-msssim`
import matplotlib.pyplot as plt


# Dataset Definition
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

        return nine_channel_input, hazy_image  # Return 9-channel input and hazy image (GT)

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


# U-Net Model Definition
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder_block = EncoderBlock(in_channels=9)  # Accept 9-channel input
        self.encoder_block_128 = EncoderBlock(in_channels=128)
        self.bottleneck = BottleneckBlock()
        self.decoder_block = DecoderBlock(in_channels=256, out_channels=128)
        self.fc_block = FullyConnectedBlock(in_channels=256, aux_channels=256)
        self.auxiliary_conv = nn.Conv2d(128, 256, kernel_size=1)
        self.final_upsample = nn.ConvTranspose2d(256, 3, kernel_size=2, stride=2)  # Output 3 channels (RGB)

    def forward(self, x):
        skip_connection1, encoded_output1 = self.encoder_block(x)
        skip_connection2, encoded_output2 = self.encoder_block_128(encoded_output1)
        skip_connections = [skip_connection1, skip_connection2]
        bottleneck_output = self.bottleneck(encoded_output2, skip_connection2)
        decoder_output = self.decoder_block(bottleneck_output, skip_connections)
        auxiliary_input_adjusted = self.auxiliary_conv(encoded_output1)
        fc_output = self.fc_block(decoder_output, auxiliary_input=auxiliary_input_adjusted)
        final_output = self.final_upsample(fc_output)
        return final_output


# Training Loop
if __name__ == "__main__":
    # Define directories
    hazy_image_dir = r'C:\Users\oishi\Documents\haze\haze\resized_input_images\resized_input_images'
    transmission_map_dir = r'C:\Users\oishi\Documents\haze\haze\predicted_transmission_maps'

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Instantiate the dataset and DataLoader
    dataset = HazyImageDataset(hazy_image_dir, transmission_map_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Instantiate the model, loss functions, and optimizer
    model = Unet()
    mse_loss = nn.MSELoss()

    def ssim_loss(predicted, target):
        return 1 - ssim(predicted, target, data_range=1.0)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training parameters
    num_epochs = 100

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for nine_channel_input, hazy_image in dataloader:
            # Move data to device (if using GPU)
            nine_channel_input = nine_channel_input.to("cpu")  # Change to "cuda" if using GPU
            hazy_image = hazy_image.to("cpu")  # Change to "cuda" if using GPU

            # Forward pass
            predicted_image = model(nine_channel_input)

            # Choose loss function based on epoch
            if epoch < 10:
                loss = mse_loss(predicted_image, hazy_image)  # Use MSE loss for Epoch < 10
            else:
                loss = ssim_loss(predicted_image, hazy_image)  # Use SSIM loss for Epoch ≥ 10

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

    # Save the trained model
    #torch.save(model.state_dict(), "unet_model_9channel.pth")
    #print("Model saved as 'unet_model_9channel.pth'.")