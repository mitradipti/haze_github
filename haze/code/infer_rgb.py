import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from trainhrm_rgb import Unet, HazyImageDataset  # Import Unet and HazyImageDataset
import matplotlib.pyplot as plt
# Define the output directory for saving predictions
output_dir = r'C:\Users\oishi\Documents\haze\haze\predicted_clear_gt_rgb'
os.makedirs(output_dir, exist_ok=True)

# Load the three trained models
model_r = Unet()
model_g = Unet()
model_b = Unet()
model_r.load_state_dict(torch.load("unet_model_hrm_clear_gt_R.pth"))
model_g.load_state_dict(torch.load("unet_model_hrm_clear_gt_G.pth"))
model_b.load_state_dict(torch.load("unet_model_hrm_clear_gt_B.pth"))
model_r.eval()
model_g.eval()
model_b.eval()
print("\nR, G, B models loaded and set to evaluation mode.")

# Define the dataset for inference
transform = transforms.Compose([
    
    transforms.ToTensor(),
])

# Define the paths for the dataset
hazy_image_dir = r'C:\Users\oishi\Documents\haze\haze\out'
transmission_map_dir = r'C:\Users\oishi\Documents\haze\haze\tmp_outr_threecha'
contrast_enhanced_dir = r'C:\Users\oishi\Documents\haze\haze\contrast_enhanced_out'
clear_gt_dir = r'C:\Users\oishi\Documents\haze\haze\gt'

dataset = HazyImageDataset(hazy_image_dir, transmission_map_dir, contrast_enhanced_dir, clear_gt_dir, transform=transform)

# Perform inference and save the predicted images
with torch.no_grad():
    for idx, (input_tensor, _, _, _) in enumerate(dataset):
        input_tensor = input_tensor.unsqueeze(0)  # [1, 9, 256, 256]

        # Inference for each channel
        output_r = model_r(input_tensor)  # [1, 1, 256, 256]
        output_g = model_g(input_tensor)  # [1, 1, 256, 256]
        output_b = model_b(input_tensor)  # [1, 1, 256, 256]

        # Resize if needed (should already be 256x256)
        output_r = F.interpolate(output_r, size=(256, 256), mode='bilinear', align_corners=False)
        output_g = F.interpolate(output_g, size=(256, 256), mode='bilinear', align_corners=False)
        output_b = F.interpolate(output_b, size=(256, 256), mode='bilinear', align_corners=False)

        # Concatenate to form RGB
        output_rgb = torch.cat([output_r, output_g, output_b], dim=1)  # [1, 3, 256, 256]
        predicted_image = output_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Normalize to [0, 1] and scale to [0, 255]
        predicted_image = (predicted_image - predicted_image.min()) / (predicted_image.max() - predicted_image.min() + 1e-8)
        predicted_image = (predicted_image * 255).astype('uint8')

        predicted_image = Image.fromarray(predicted_image)
        output_path = os.path.join(output_dir, f"predicted_{idx + 1}.jpg")
        predicted_image.save(output_path)
        print(f"Saved predicted image: {output_path}")