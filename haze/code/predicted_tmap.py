import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from blocks import EncoderBlock, BottleneckBlock, DecoderBlock, FullyConnectedBlock
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        return final_output

# Prediction Pipeline
if __name__ == "__main__":
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the trained model parameters
    model = Unet()
    model.load_state_dict(torch.load("unet_model_out.pth"))  # Load the saved parameters
    model.eval()  # Set the model to evaluation mode
    print("\nModel parameters loaded and set to evaluation mode.")

    # Directory containing hazy images
    hazy_image_dir = r"C:\Users\oishi\Documents\haze\haze\out"
    output_dir = r"C:\Users\oishi\Documents\haze\haze\tmp_outr_predicted"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all hazy images in the directory
    for hazy_image_filename in os.listdir(hazy_image_dir):
        hazy_image_path = os.path.join(hazy_image_dir, hazy_image_filename)
        hazy_image = Image.open(hazy_image_path).convert('RGB')
        hazy_image_tensor = transform(hazy_image).unsqueeze(0)  # Add batch dimension

        # Predict the transmission map
        with torch.no_grad():
            predicted_map = model(hazy_image_tensor)  # Shape: [1, 1, H, W]
        
        # Upsample the predicted map to 1600x1200
        predicted_map_upsampled = F.interpolate(predicted_map, size=(1200, 1600), mode='bilinear', align_corners=False)
        # Convert the predicted map to a NumPy array
        predicted_map_np = predicted_map_upsampled.squeeze(0).squeeze(0).numpy()  # Remove batch and channel dimensions

        # Normalize the predicted map for visualization
        predicted_map_np = (predicted_map_np - predicted_map_np.min()) / (predicted_map_np.max() - predicted_map_np.min())
        predicted_map_np_255 = (predicted_map_np * 255).astype('uint8')

        # Save the upsampled predicted transmission map
        output_path = os.path.join(output_dir, f"predicted_{hazy_image_filename}")
        predicted_map_image = Image.fromarray(predicted_map_np_255)
        predicted_map_image.save(output_path)
        print(f"Predicted transmission map saved as '{output_path}'.")