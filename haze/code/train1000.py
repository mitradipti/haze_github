import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from blocks import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
import torch.nn.functional as F

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
        hazy_image = Image.open(hazy_image_path).convert('RGB')
        transmission_map = Image.open(transmission_map_path).convert('L')
        if self.transform:
            hazy_image = self.transform(hazy_image)
            transmission_map = self.transform(transmission_map)
        return hazy_image, transmission_map

# U-Net Model Definition
class Unet(nn.Module):
    def __init__(self, version="2M"):
        super(Unet, self).__init__()
        if version == "2M":
            bottleneck_channels = 256
        elif version == "1M":
            bottleneck_channels = 128
        else:
            raise ValueError("Invalid version. Choose '2M' or '1M'.")

        # Update EncoderBlock to match its constructor
        self.encoder_block = EncoderBlock(in_channels=3)  # First block takes 3 input channels (RGB image)
        self.encoder_block_128 = EncoderBlock(in_channels=128)  # Second block takes 128 input channels

        # Bottleneck and DecoderBlock remain unchanged
        self.bottleneck = BottleneckBlock()
        self.decoder_block = DecoderBlock(in_channels=bottleneck_channels, out_channels=64)
        self.fc_block = FullyConnectedBlock(in_channels=bottleneck_channels, aux_channels=bottleneck_channels)

        # Fix auxiliary_conv to match the output of the first EncoderBlock
        self.auxiliary_conv = nn.Conv2d(128, bottleneck_channels, kernel_size=1)

    def forward(self, x):
        skip_connection1, encoded_output1 = self.encoder_block(x)  # Output: [batch_size, 128, H/2, W/2]
        skip_connection2, encoded_output2 = self.encoder_block_128(encoded_output1)  # Output: [batch_size, 128, H/4, W/4]
        skip_connections = [skip_connection1, skip_connection2]
        bottleneck_output = self.bottleneck(encoded_output2, skip_connection2)
        decoder_output = self.decoder_block(bottleneck_output, skip_connections)
        auxiliary_input_adjusted = self.auxiliary_conv(encoded_output1)
        fc_output = self.fc_block(decoder_output, auxiliary_input=auxiliary_input_adjusted)
        final_output = F.interpolate(fc_output, size=(256, 256), mode='bilinear', align_corners=False)
        return final_output

# Training Pipeline
if __name__ == "__main__":
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 1280x720
        transforms.ToTensor(),
    ])
    hazy_image_dir = r'C:\Users\oishi\Documents\haze\haze\outdoor'
    transmission_map_dir = r'C:\Users\oishi\Documents\haze\haze\tmp_ntire_outdoor_r_updated'

    dataset = HazyImageDataset(hazy_image_dir, transmission_map_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model, Loss, and Optimizer
    model = Unet(version="2M")  # Choose "2M" or "1M"
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.96)

    # Inspect Model Parameters
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}")

    # Calculate Total Number of Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Number of Parameters: {total_params}")

    # Training Loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            # Ensure inputs and labels are on the CPU
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Step the scheduler
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

    # Save the trained model
    torch.save(model.state_dict(), "unet_model_updated_ntire.pth")
    print("\nModel saved as 'unet_model_updated_ntire.pth'.")