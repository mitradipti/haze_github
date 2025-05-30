import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from blockshrm import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
import torch.nn.functional as F
#from pytorch_msssim import ssim

# MSE Loss
mse_loss = torch.nn.MSELoss()

# SSIM Loss
#def ssim_loss(predicted, label):
#    return 1 - ssim(predicted, label, data_range=1.0)

# Dataset Definition
class HazyImageDataset(Dataset):
    def __init__(self, hazy_image_dir, transmission_map_dir, contrast_enhanced_dir, clear_gt_dir, transform=None):
        self.hazy_image_dir = hazy_image_dir
        self.transmission_map_dir = transmission_map_dir
        self.contrast_enhanced_dir = contrast_enhanced_dir
        self.clear_gt_dir = clear_gt_dir
        
        self.transform = transform
        self.hazy_image_filenames = os.listdir(hazy_image_dir)

    def __len__(self):
        return len(self.hazy_image_filenames)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_image_dir, self.hazy_image_filenames[idx])
        transmission_map_path = os.path.join(self.transmission_map_dir, self.hazy_image_filenames[idx])
        contrast_enhanced_path = os.path.join(self.contrast_enhanced_dir, self.hazy_image_filenames[idx])
        clear_gt_path = os.path.join(self.clear_gt_dir, self.hazy_image_filenames[idx])
        
        hazy_image = Image.open(hazy_image_path).convert('RGB')
        transmission_map = Image.open(transmission_map_path).convert('L')
        contrast_enhanced_image = Image.open(contrast_enhanced_path).convert('RGB')
        clear_gt_image = Image.open(clear_gt_path).convert('RGB')

        if self.transform:
            hazy_image = self.transform(hazy_image)
            transmission_map = self.transform(transmission_map)
            contrast_enhanced_image = self.transform(contrast_enhanced_image)
            clear_gt_image = self.transform(clear_gt_image)

            # Normalize the hazy image explicitly to [0, 1]
            hazy_image = (hazy_image - hazy_image.min()) / (hazy_image.max() - hazy_image.min())

            # Normalize the transmission map explicitly to [0, 1]
            transmission_map = (transmission_map - transmission_map.min()) / (transmission_map.max() - transmission_map.min())
            
            # Normalize the contrast-enhanced image explicitly to [0, 1]
            contrast_enhanced_image = (contrast_enhanced_image - contrast_enhanced_image.min()) / (contrast_enhanced_image.max() - contrast_enhanced_image.min())
        
        # Normalize the ground truth image explicitly to [0, 1]
            clear_gt_image = (clear_gt_image - clear_gt_image.min()) / (clear_gt_image.max() - clear_gt_image.min())

        # Debugging: Check the range of the hazy image
        #print(f"Hazy Image Range After Normalization: {hazy_image.min().item()} to {hazy_image.max().item()}")
        
        #print(f"Normalized Transmission Map Range: {transmission_map.min().item()} to {transmission_map.max().item()}")
        
        # Debugging: Check the range of the ground truth image
        #print(f"Ground Truth Image Range After Normalization: {clear_gt_image.min().item()} to {clear_gt_image.max().item()}")
        
        # Debugging: Check the range of the contrast-enhanced image
        #print(f"Contrast Enhanced Image Range After Normalization: {contrast_enhanced_image.min().item()} to {contrast_enhanced_image.max().item()}")
        # Debugging: Check the range of the ground truth image
        #print(f"Ground Truth Image Range: {clear_gt_image.min().item()} to {clear_gt_image.max().item()}")
        
        #print(f"Transmission Map Range: {transmission_map.min().item()} to {transmission_map.max().item()}")
        # Create a 9-channel input tensor by concatenating the images
        combined_input = torch.cat([hazy_image, transmission_map.repeat(3, 1, 1), contrast_enhanced_image], dim=0)

        #print(f"Combined Input Range: {combined_input.min().item()} to {combined_input.max().item()}")
        
        # Return the 9-channel input and the clear image as the label
        return combined_input, clear_gt_image

# U-Net Model Definition
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder_block = EncoderBlock(in_channels=9)
        self.encoder_block_128 = EncoderBlock(in_channels=128)
        self.bottleneck = BottleneckBlock()
        self.decoder_block = DecoderBlock(in_channels=256, out_channels=128)
        self.fc_block = FullyConnectedBlock(in_channels=256, aux_channels=256)
        self.auxiliary_conv = nn.Conv2d(128, 256, kernel_size=1)

    def forward(self, x):
        skip_connection1, encoded_output1 = self.encoder_block(x)
        skip_connection2, encoded_output2 = self.encoder_block_128(encoded_output1)
        skip_connections = [skip_connection1, skip_connection2]
        bottleneck_output = self.bottleneck(encoded_output2, skip_connection2)
        decoder_output = self.decoder_block(bottleneck_output, skip_connections)
        auxiliary_input_adjusted = self.auxiliary_conv(encoded_output1)
        fc_output = self.fc_block(decoder_output, auxiliary_input=auxiliary_input_adjusted)
        return fc_output

# Training Pipeline
if __name__ == "__main__":
    # Dataset and DataLoader (Directory Dimensions: 1600x1200)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    hazy_image_dir = r'C:\Users\oishi\Documents\haze\haze\out'
    transmission_map_dir = r'C:\Users\oishi\Documents\haze\haze\tmp_outr_threecha'
    contrast_enhanced_dir = r'C:\Users\oishi\Documents\haze\haze\contrast_enhanced_out'
    clear_gt_dir = r'C:\Users\oishi\Documents\haze\haze\gt'

    dataset = HazyImageDataset(hazy_image_dir, transmission_map_dir, contrast_enhanced_dir, clear_gt_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model, Loss, and Optimizer
    model = Unet()
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
            outputs = model(inputs)
            
            # Resize the model's output to match the ground truth size
            outputs_resized = F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)
            
            # Select the loss function
            loss = mse_loss(outputs_resized, labels)  # Use MSE Loss
            # if epoch < 10:
            #     loss = mse_loss(outputs_resized, labels)  # Use MSE Loss
            # else:
            #     loss = ssim_loss(outputs_resized, labels)  # Use SSIM Loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

    # Save the trained model
    torch.save(model.state_dict(), "unet_model_hrm_clear_gt_scaled.pth")
    print("\nModel saved as 'unet_model_hrm_clear_gt_scaled.pth'.")

    # Load the trained model
    model.load_state_dict(torch.load("unet_model_hrm_clear_gt_scaled.pth"))
    model.eval()  # Set the model to evaluation mode
    print("\nModel loaded and set to evaluation mode.")