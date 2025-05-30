#Testing

# Define Encoder Block 0 with 9 input channels
encoder_block_0 = EncoderBlock(in_channels=9)

# Example input with 9 channels
input_tensor = torch.randn(1, 9, 256, 256)  # Batch size = 1, 9 channels, 256x256 resolution

# Pass through Encoder Block 0
encoder_output, _ = encoder_block_0(input_tensor)

print("Encoder Output Shape:", encoder_output.shape)

# Define Auxiliary Input
auxiliary_input = torch.randn(1, 256, 128, 128)  # Auxiliary input for FC block

# Define Fully Connected Block
fc_block = FullyConnectedBlock(in_channels=256, aux_channels=256)

# Define Transition Layer
transition_layer = nn.Conv2d(128, 256, kernel_size=1)  # 1x1 convolution to increase channels

# Pass through Transition Layer
encoder_output = transition_layer(encoder_output)  # Adjust channels to match FC block input

# Pass through Fully Connected Block
fc_output = fc_block(encoder_output, auxiliary_input)

print("Output Shape:", fc_output.shape)  # Expected: (1, 3, 256, 256)

# Example combined input tensor (prepared earlier)
# Shape: (batch_size, 9, height, width)
input_tensor = combined_input  # From prepare_combined_input function

# Initialize the network components
encoder_block = EncoderBlock(in_channels=9)  # 9 input channels
bottleneck_block = BottleneckBlock()
decoder_block = DecoderBlock(in_channels=256, out_channels=256)
fc_block = FullyConnectedBlock(in_channels=256, aux_channels=256)

# Example input tensor (prepared earlier)
input_tensor = torch.randn(1, 9, 256, 256)  # Batch size = 1, 9 channels, 256x256 resolution

# Pass the input through the Encoder Block
print("Passing through Encoder Block...")
skip_connection, encoder_output = encoder_block(input_tensor)
print("Encoder Output Shape:", encoder_output.shape)  # Expected: (batch_size, 128, height/2, width/2)

# Pass the encoder output and skip connection through the Bottleneck Block
print("Passing through Bottleneck Block...")
bottleneck_output = bottleneck_block(encoder_output, skip_connection)
print("Bottleneck Output Shape:", bottleneck_output.shape)  # Expected: (batch_size, 256, height/2, width/2)

# Pass the bottleneck output and skip connection through the Decoder Block
print("Passing through Decoder Block...")
decoder_output = decoder_block(bottleneck_output, [skip_connection])
print("Decoder Output Shape:", decoder_output.shape)  # Expected: (batch_size, 128, height, width)

# Auxiliary input for the Fully Connected Block
auxiliary_input = torch.randn(1, 256, decoder_output.shape[2], decoder_output.shape[3])  # Example auxiliary input

# Pass the decoder output and auxiliary input through the Fully Connected Block
print("Passing through Fully Connected Block...")
fc_output = fc_block(decoder_output, auxiliary_input)
print("Final Output Shape:", fc_output.shape)  # Expected: (batch_size, 3, height, width)  # Expected: (batch_size, 3, height, width)




import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from blockshrm import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_msssim import ssim

# MSE Loss
mse_loss = torch.nn.MSELoss()

# SSIM Loss
def ssim_loss(predicted, label):
    return 1 - ssim(predicted, label, data_range=1.0)  

# Dataset Definition
class HazyImageDataset(Dataset):
    def __init__(self, hazy_image_dir, transmission_map_dir, contrast_enhanced_dir, transform=None):
        self.hazy_image_dir = hazy_image_dir
        self.transmission_map_dir = transmission_map_dir
        self.contrast_enhanced_dir = contrast_enhanced_dir
        #self.clear_gt_dir = clear_gt_dir
        self.transform = transform
        self.hazy_image_filenames = os.listdir(hazy_image_dir)
 
    def __len__(self):
        return len(self.hazy_image_filenames)
 
    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_image_dir, self.hazy_image_filenames[idx])
        transmission_map_path = os.path.join(self.transmission_map_dir, self.hazy_image_filenames[idx])
        contrast_enhanced_path = os.path.join(self.contrast_enhanced_dir, self.hazy_image_filenames[idx])
        #clear_gt_path = os.path.join(self.clear_gt_dir, self.hazy_image_filenames[idx])
        
        hazy_image = Image.open(hazy_image_path).convert('RGB')
        transmission_map = Image.open(transmission_map_path).convert('L')
        contrast_enhanced_image = Image.open(contrast_enhanced_path).convert('RGB')
        #clear_gt_image = Image.open(clear_gt_path).convert('L')
        
        if self.transform:
            hazy_image = self.transform(hazy_image)
            transmission_map = self.transform(transmission_map)
            contrast_enhanced_image = self.transform(contrast_enhanced_image)
            #clear_gt_image = self.transform(clear_gt_image)
        return hazy_image, transmission_map, contrast_enhanced_image  # Return contrast-enhanced image and clear GT image as well
 
# U-Net Model Definition
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder_block = EncoderBlock(in_channels=9)
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
        #final_output = F.interpolate(fc_output, size=(256, 256), mode='bilinear', align_corners=False)
        return fc_output
 
# Training Pipeline
if __name__ == "__main__":
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    hazy_image_dir = r'C:\Users\oishi\Documents\haze\haze\out'
    transmission_map_dir = r'C:\Users\oishi\Documents\haze\haze\tmp_outr_predicted'
    contrast_enhanced_dir = r'C:\Users\oishi\Documents\haze\haze\contrast_enhanced_out'
    #clear_gt_dir = r'C:\Users\oishi\Documents\haze\haze\tmp_out_clear'
    
    dataset = HazyImageDataset(hazy_image_dir, transmission_map_dir, contrast_enhanced_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
 
    # Model, Loss, and Optimizer
    model = Unet()
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
    # Inspect Model Parameters
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Shape: {param.shape}")
 
    # Calculate Total Number of Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Number of Parameters: {total_params}")
 
    # Training Loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            # Ensure inputs and labels are on the CPU
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            optimizer.zero_grad()
            outputs = model(inputs)
            
            #loss = criterion(outputs, labels)
            # Select the loss function
        if epoch < 10:
            loss = mse_loss(outputs, labels)  # Use MSE Loss
        else:
            loss = ssim_loss(outputs, labels)  # Use SSIM Loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')
 
    # Save the trained model
    torch.save(model.state_dict(), "unet_model_hrm.pth")
    print("\nModel saved as 'unet_model_hrm.pth'.")
 
    # Load the trained model
    model.load_state_dict(torch.load("unet_model_hrm.pth"))
    model.eval()  # Set the model to evaluation mode
    print("\nModel loaded and set to evaluation mode.")
 