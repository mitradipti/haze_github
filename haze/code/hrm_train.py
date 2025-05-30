import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from blockshrm import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
import torch
import torch.nn as nn
from pytorch_msssim import SSIM  # Import SSIM loss if using pytorch-msssim

# Custom Dataset Class
class HazeDataset(Dataset):
    def __init__(self, hazy_dir, clean_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.hazy_images = sorted(os.listdir(hazy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))

        # Sanity check: Ensure both directories have the same number of images
        if len(self.hazy_images) != len(self.clean_images):
            raise ValueError("Mismatch between hazy and clean image counts. "
                             f"Hazy: {len(self.hazy_images)}, Clean: {len(self.clean_images)}")

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])

        hazy_image = Image.open(hazy_path).convert("RGB")
        clean_image = Image.open(clean_path).convert("RGB")

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clean_image = self.transform(clean_image)

        return hazy_image, clean_image


# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),         # Convert images to PyTorch tensors
])

# Initialize Dataset and DataLoader
hazy_dir = r"C:\Users\oishi\Documents\haze\haze\testing\hazy_700"
clean_dir = r"C:\Users\oishi\Documents\haze\haze\testing\clear_700"
dataset = HazeDataset(hazy_dir, clean_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize network components
encoder_block = EncoderBlock(in_channels=9)
bottleneck_block = BottleneckBlock()
decoder_block = DecoderBlock(in_channels=256, out_channels=128)
fc_block = FullyConnectedBlock(in_channels=256, aux_channels=256)

# Define loss functions
mse_loss = nn.MSELoss()
ssim_loss = SSIM(data_range=1.0)  # SSIM loss from pytorch-msssim

# Define optimizer
optimizer = torch.optim.Adam(
    list(encoder_block.parameters()) +
    list(bottleneck_block.parameters()) +
    list(decoder_block.parameters()) +
    list(fc_block.parameters()),
    lr=0.001
)

# Number of epochs for training
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    for hazy_images, clean_images in dataloader:
        # Forward pass
        skip_connection, encoder_output = encoder_block(hazy_images)
        bottleneck_output = bottleneck_block(encoder_output, skip_connection)
        decoder_output = decoder_block(bottleneck_output, [skip_connection])
        auxiliary_input = torch.randn(hazy_images.size(0), 256, decoder_output.shape[2], decoder_output.shape[3])
        output = fc_block(decoder_output, auxiliary_input)

        # Compute loss
        if epoch < 10:
            loss = mse_loss(output, clean_images)  # Use MSE loss for the first 10 epochs
        else:
            loss = 1 - ssim_loss(output, clean_images)  # Use SSIM loss for subsequent epochs

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")