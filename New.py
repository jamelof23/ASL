import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import subprocess

# # Clone the InterFaceGAN repository
# if not os.path.exists("interfacegan"):
#     print("Cloning the InterFaceGAN repository...")
#     subprocess.run(["git", "clone", "https://github.com/genforce/interfacegan.git"], check=True)


# Add InterFaceGAN to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
interfacegan_path = os.path.join(script_dir, 'interfacegan-master')
if interfacegan_path not in sys.path:
    sys.path.append(interfacegan_path)


from models.stylegan_generator import StyleGANGenerator

# Define device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a random seed for reproducibility across CPU, NumPy, and GPU (CUDA)
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Model name
model_name = 'stylegan_ffhq'  # Ensure this model exists in MODEL_POOL

def load_interfacegan_generator(model_name):
    # Initialize the InterFaceGAN generator
    generator = StyleGANGenerator(model_name)
    # Move the generator to the desired device
    generator.model.to(device)
    return generator

# Load the InterFaceGAN generator
generator = load_interfacegan_generator(model_name)

# Check for multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    # Wrap the generator model with DataParallel
    generator.model = nn.DataParallel(generator.model)

# Ensure the generator is moved to GPU(s)
generator.model.to(device)

# Print confirmation
print(f"InterFaceGAN generator is loaded on device: {next(generator.model.parameters()).device}")

# VGG16 layers to capture high-frequency details
vgg16_layers = [1, 3, 6, 8, 11, 15, 22, 29]

# Define VGG16 feature extractor
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, layers):
        super(VGG16FeatureExtractor, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:max(layers)+1]
        self.layers = layers

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.vgg16):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

# Initialize VGG16 feature extractor
vgg16_extractor = VGG16FeatureExtractor(layers=vgg16_layers).to(device).eval()  # Set to evaluation mode

# Preprocess the input image to match VGG16 requirements
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for VGG16
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])

# Load input image and extract VGG16 features
image_path = '1.png'  # Update with your target image path
input_image = Image.open(image_path).convert('RGB')
input_image_tensor = preprocess(input_image).unsqueeze(0).to(device)

# Extracting target features
with torch.no_grad():
    target_features = vgg16_extractor(input_image_tensor)
print("Extracted target features from VGG16 layers.")

# Sample a random latent vector in Z space
latent_code = torch.randn(1, 512, device=device, requires_grad=True)

# Create an optimizer for optimized_ws_tensor
optimizer = optim.Adam([latent_code], lr=0.01)



# Training process
num_iterations = 300
layer_weights = [1.0] * len(vgg16_layers)  # Equal weights for each VGG16 layer

for iteration in range(num_iterations):
    optimizer.zero_grad()  # Clear previous gradients

    # Map the latent code to W space
    w = generator.model.mapping(latent_code)  # Shape: [1, 512]

    # Get the number of style inputs required by the synthesis network
    num_ws = generator.num_layers  # Access num_layers from generator

    # Repeat w to have shape [batch_size, num_ws, w_dim]
    w = w.unsqueeze(1).repeat(1, num_ws, 1)  # Shape: [1, num_ws, 512]

    # Generate image with the current w
    image = generator.model.synthesis(w)

    # The generated image is in [-1, 1], so scale it to [0, 1]
    generated_image = (image + 1) / 2.0

    # Resize the generated image for VGG16 feature extraction
    generated_image_resized = F.interpolate(
        generated_image, size=(224, 224), mode='bilinear', align_corners=False
    )

    # Extract features from the generated image using VGG16
    generated_features = vgg16_extractor(generated_image_resized)

    # Compute losses and backpropagate
    losses = []
    total_loss = 0
    for i in range(len(vgg16_layers)):
        # Compute the MSE loss for the current layer
        loss = F.mse_loss(generated_features[i], target_features[i])
        weighted_loss = layer_weights[i] * loss
        total_loss += weighted_loss
        losses.append(loss.item())

    # Calculate intra-channel color diversity loss
    generated_image_np = generated_image.squeeze(0).permute(1, 2, 0)  # Shape: (H, W, C)
    red_channel = generated_image_np[:, :, 0]
    green_channel = generated_image_np[:, :, 1]
    blue_channel = generated_image_np[:, :, 2]

    # Calculate diversity loss between channels
    color_diversity_loss = (
        F.mse_loss(red_channel, green_channel) +
        F.mse_loss(red_channel, blue_channel) +
        F.mse_loss(green_channel, blue_channel)
    )

    # Add color diversity loss to total loss
    total_loss += 2.5 * color_diversity_loss  # Adjust weight as needed

    # Print initial loss
    if iteration == 0:
        print("Initial losses for each VGG16 layer:")
        for idx, loss_value in enumerate(losses):
            print(f"Layer {vgg16_layers[idx]}: {loss_value:.6f}")
        print(f"Initial color diversity loss: {color_diversity_loss.item():.6f}")

    total_loss.backward()
    optimizer.step()

    # Print losses every 100 iterations
    if iteration % 100 == 0 or iteration == num_iterations - 1:
        loss_str = ', '.join(
            [f"Layer {vgg16_layers[i]}: {losses[i]:.4f}" for i in range(len(losses))]
        )
        print(
            f"Iteration {iteration}, Total Loss: {total_loss.item():.4f}, "
            f"Color Diversity Loss: {color_diversity_loss.item():.4f}"
        )


# Convert the final generated image to numpy format
final_generated_image_np = generated_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

# Clip the values to ensure they are within [0, 1]
final_generated_image_np = np.clip(final_generated_image_np, 0, 1)

# Save the final generated image
save_dir = "/data/"
if os.path.exists(save_dir):
    final_image_path = os.path.join(save_dir, "image215.png")
else:
    final_image_path = os.path.join(script_dir, "image215.png")

# Save the image using clipped values
plt.imsave(final_image_path, final_generated_image_np)
print(f"Generated image saved to {final_image_path}")

# Display the input and generated images using matplotlib
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(input_image)
axs[0].set_title('Input Image')
axs[0].axis('off')

axs[1].imshow(final_generated_image_np)
axs[1].set_title('Generated Image')
axs[1].axis('off')

plt.show()

