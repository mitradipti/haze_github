import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from trainhrm import Unet, HazyImageDataset  # Import Unet and HazyImageDataset
import matplotlib.pyplot as plt
# Define the output directory for saving predictions
output_dir = r'C:\Users\oishi\Documents\haze\haze\predicted_clear_gttest_scaled'
os.makedirs(output_dir, exist_ok=True)

# Load the trained model
model = Unet()
model.load_state_dict(torch.load("unet_model_hrm_clear_gt_scaled.pth"))
model.eval()  # Set the model to evaluation mode
print("\nModel loaded and set to evaluation mode.")

# Define the dataset for inference
transform = transforms.Compose([
    transforms.Resize((256, 256)),
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
    for idx, (input_tensor, _) in enumerate(dataset):
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        # Perform inference
        predicted_output = model(input_tensor)

        # Debugging: Print the shape and range of the output tensor
        print(f"Predicted Output Shape: {predicted_output.shape}")
        print(f"Predicted Output Min: {predicted_output.min().item()}, Max: {predicted_output.max().item()}")

        # Resize the predicted output to match the input size
        predicted_output_resized = F.interpolate(predicted_output, size=(256, 256), mode='bilinear', align_corners=False)

        # Convert the predicted output to an image
        predicted_image = predicted_output_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Normalize the predicted output to the range [0, 1]
        predicted_image = (predicted_image - predicted_image.min()) / (predicted_image.max() - predicted_image.min())
        predicted_image = (predicted_image * 255).astype('uint8')  # Scale to [0, 255]
        
        #plt.imshow(predicted_image)
        #plt.axis("off")
        #plt.title(f"Predicted Image {idx + 1}")
        #plt.show()
        predicted_image = Image.fromarray(predicted_image)

        # Save the predicted image
        output_path = os.path.join(output_dir, f"predicted_{idx + 1}.jpg")
        predicted_image.save(output_path)
        print(f"Saved predicted image: {output_path}")