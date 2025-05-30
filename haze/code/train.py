import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from blocks import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
import matplotlib.pyplot as plt
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
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder_block = EncoderBlock(in_channels=3)
        self.encoder_block_128 = EncoderBlock(in_channels=128)
        self.bottleneck = BottleneckBlock()
        self.decoder_block = DecoderBlock(in_channels=256, out_channels=128)
        self.fc_block = FullyConnectedBlock(in_channels=256, aux_channels=256)
        self.auxiliary_conv = nn.Conv2d(128, 256, kernel_size=1)  # Moved from forward to init
        

    def forward(self, x):
        skip_connection1, encoded_output1 = self.encoder_block(x)
        skip_connection2, encoded_output2 = self.encoder_block_128(encoded_output1)
        skip_connections = [skip_connection1, skip_connection2]
        bottleneck_output = self.bottleneck(encoded_output2, skip_connection2)  # Fixed method call
        decoder_output = self.decoder_block(bottleneck_output, skip_connections)
        auxiliary_input_adjusted = self.auxiliary_conv(encoded_output1)  # Use pre-defined layer
        fc_output = self.fc_block(decoder_output, auxiliary_input=auxiliary_input_adjusted)
    # Upsample the output to match the ground truth size (256x256)
        final_output = F.interpolate(fc_output, size=(256, 256), mode='bilinear', align_corners=False)
        #upsampled_output = F.interpolate(final_output, size=(1600, 1200), mode='bilinear', align_corners=False)
        return final_output

# Training Pipeline
if __name__ == "__main__":
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    hazy_image_dir = r'C:\Users\oishi\Documents\haze\haze\out'
    transmission_map_dir = r'C:\Users\oishi\Documents\haze\haze\tmp_out'

    dataset = HazyImageDataset(hazy_image_dir, transmission_map_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model, Loss, and Optimizer
    model = Unet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Inspect Model Parameters
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}")

    # Calculate Total Number of Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Number of Parameters: {total_params}")

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            # Ensure inputs and labels are on the CPU
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            optimizer.zero_grad()
            final_output, _ = model(inputs)
            loss = criterion(final_output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

    # Save the trained model
    torch.save(model.state_dict(), "unet_model_outn.pth")
    print("\nModel saved as 'unet_model_outn.pth'.")

    # Load the trained model
    model.load_state_dict(torch.load("unet_model_outn.pth"))
    model.eval()  # Set the model to evaluation mode
    print("\nModel loaded and set to evaluation mode.")