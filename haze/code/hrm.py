import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from blockshrm import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
from pytorch_msssim import SSIM  # For SSIM loss

# Custom Dataset Class
class HazeDataset(Dataset):
    def __init__(self, original_dir, contrast_dir, transmission_dir, transform=None):
        self.original_dir = original_dir
        self.contrast_dir = contrast_dir
        self.transmission_dir = transmission_dir
        self.transform = transform

        # Get all filenames
        self.original_images = sorted(os.listdir(original_dir))
        self.contrast_images = sorted(os.listdir(contrast_dir))
        self.transmission_maps = sorted(os.listdir(transmission_dir))

        # Sanity check: Ensure all directories have the same number of images
        if not (len(self.original_images) == len(self.contrast_images) == len(self.transmission_maps)):
            raise ValueError("Mismatch between the number of images in the directories.")

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        # Load original input image
        original_path = os.path.join(self.original_dir, self.original_images[idx])
        original_image = Image.open(original_path).convert("RGB")

        # Load contrast-enhanced image
        contrast_path = os.path.join(self.contrast_dir, self.contrast_images[idx])
        contrast_image = Image.open(contrast_path).convert("RGB")

        # Load transmission map
        transmission_path = os.path.join(self.transmission_dir, self.transmission_maps[idx])
        transmission_map = Image.open(transmission_path).convert("RGB")  # 3-channel transmission map

        # Apply transformations
        if self.transform:
            original_image = self.transform(original_image)
            contrast_image = self.transform(contrast_image)
            transmission_map = self.transform(transmission_map)

        # Concatenate the three 3-channel images to form a 9-channel input tensor
        input_tensor = torch.cat([original_image, contrast_image, transmission_map], dim=0)

        return input_tensor


# Define transformations
transform = transforms.Compose([
    
    transforms.ToTensor(),         # Convert images to PyTorch tensors
])

# Paths to directories
original_dir = r"C:\Users\oishi\Documents\haze\haze\resized_input_images\resized_input_images"
contrast_dir = r"C:\Users\oishi\Documents\haze\haze\contrast_enhanced_images"
transmission_dir = r"C:\Users\oishi\Documents\haze\haze\three_channel_transmission_maps"

# Initialize the dataset and dataloader
dataset = HazeDataset(original_dir, contrast_dir, transmission_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize network components
encoder_block = EncoderBlock(in_channels=9)
bottleneck_block = BottleneckBlock()
decoder_block = DecoderBlock(in_channels=256, out_channels=128)
fc_block = FullyConnectedBlock(in_channels=256, aux_channels=256)

# Define optimizer
optimizer = torch.optim.Adam(
    list(encoder_block.parameters()) +
    list(bottleneck_block.parameters()) +
    list(decoder_block.parameters()) +
    list(fc_block.parameters()),
    lr=0.001
)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for input_tensor in dataloader:
        # Forward pass
        skip_connection, encoder_output = encoder_block(input_tensor)
        bottleneck_output = bottleneck_block(encoder_output, skip_connection)
        decoder_output = decoder_block(bottleneck_output, [skip_connection])
        auxiliary_input = torch.randn(input_tensor.size(0), 256, decoder_output.shape[2], decoder_output.shape[3])
        output = fc_block(decoder_output, auxiliary_input)

        # Save the output images for visual inspection
        output_images = output.permute(0, 2, 3, 1).cpu().detach().numpy()  # Convert to (batch_size, height, width, channels)
        for i, img in enumerate(output_images):
            img = (img * 255).astype("uint8")  # Convert to uint8
            Image.fromarray(img).save(f"output_epoch{epoch+1}_img{i+1}.png")

    print(f"Epoch [{epoch+1}/{num_epochs}] completed.")

# Save the trained model
torch.save({
    'encoder_block': encoder_block.state_dict(),
    'bottleneck_block': bottleneck_block.state_dict(),
    'decoder_block': decoder_block.state_dict(),
    'fc_block': fc_block.state_dict(),
}, 'hrm_model.pth')