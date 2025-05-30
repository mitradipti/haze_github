import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from blocksrgb import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
import torch.nn.functional as F

# MSE Loss
mse_loss = torch.nn.MSELoss()

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

            # Normalize all images to [0, 1]
            hazy_image = (hazy_image - hazy_image.min()) / (hazy_image.max() - hazy_image.min())
            transmission_map = (transmission_map - transmission_map.min()) / (transmission_map.max() - transmission_map.min())
            contrast_enhanced_image = (contrast_enhanced_image - contrast_enhanced_image.min()) / (contrast_enhanced_image.max() - contrast_enhanced_image.min())
            clear_gt_image = (clear_gt_image - clear_gt_image.min()) / (clear_gt_image.max() - clear_gt_image.min())

        combined_input = torch.cat([hazy_image, transmission_map.repeat(3, 1, 1), contrast_enhanced_image], dim=0)
        # Return each channel separately for R, G, B (Separate single channel tensor)
        return combined_input, clear_gt_image[0:1, :, :], clear_gt_image[1:2, :, :], clear_gt_image[2:3, :, :]

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
    # Dataset and DataLoader
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

    # Instantiate three separate models and optimizers for R, G, B
    model_r = Unet()
    model_g = Unet()
    model_b = Unet()
    optimizer_r = optim.Adam(model_r.parameters(), lr=0.0001)
    optimizer_g = optim.Adam(model_g.parameters(), lr=0.0001)
    optimizer_b = optim.Adam(model_b.parameters(), lr=0.0001)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model_r.train()
        model_g.train()
        model_b.train()
        running_loss_r = 0.0
        running_loss_g = 0.0
        running_loss_b = 0.0
        for inputs, label_r, label_g, label_b in dataloader:
            # R channel
            optimizer_r.zero_grad()
            output_r = model_r(inputs)
            output_r = F.interpolate(output_r, size=label_r.shape[2:], mode='bilinear', align_corners=False)
            loss_r = mse_loss(output_r, label_r)
            loss_r.backward()
            optimizer_r.step()
            running_loss_r += loss_r.item()

            # G channel
            optimizer_g.zero_grad()
            output_g = model_g(inputs)
            output_g = F.interpolate(output_g, size=label_g.shape[2:], mode='bilinear', align_corners=False)
            loss_g = mse_loss(output_g, label_g)
            loss_g.backward()
            optimizer_g.step()
            running_loss_g += loss_g.item()

            # B channel
            optimizer_b.zero_grad()
            output_b = model_b(inputs)
            output_b = F.interpolate(output_b, size=label_b.shape[2:], mode='bilinear', align_corners=False)
            loss_b = mse_loss(output_b, label_b)
            loss_b.backward()
            optimizer_b.step()
            running_loss_b += loss_b.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss R: {running_loss_r/len(dataloader)}, Loss G: {running_loss_g/len(dataloader)}, Loss B: {running_loss_b/len(dataloader)}')

    # Save the trained models
    torch.save(model_r.state_dict(), "unet_model_hrm_clear_gt_R.pth")
    torch.save(model_g.state_dict(), "unet_model_hrm_clear_gt_G.pth")
    torch.save(model_b.state_dict(), "unet_model_hrm_clear_gt_B.pth")
    print("\nModels saved as 'unet_model_hrm_clear_gt_R/G/B.pth'.")