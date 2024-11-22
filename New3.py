import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Add InterFaceGAN to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
interfacegan_path = os.path.join(script_dir, 'interfacegan-master')
if interfacegan_path not in sys.path:
    sys.path.append(interfacegan_path)

from models.stylegan_generator import StyleGANGenerator

# Define device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Model name
model_name = 'stylegan_ffhq'  # Ensure this model exists in MODEL_POOL

# Initialize the InterFaceGAN generator
generator = StyleGANGenerator(model_name)
generator.model.to(device)

# Check for multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    generator.model = nn.DataParallel(generator.model)

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
image_path = '7.png'  # Update with your target image path
input_image = Image.open(image_path).convert('RGB')
input_image_tensor = preprocess(input_image).unsqueeze(0).to(device)

# Extract target features from the input image
with torch.no_grad():
    target_features = vgg16_extractor(input_image_tensor)
print("Extracted target features from VGG16 layers.")

# Sample a random latent vector in Z space and map it to W space
with torch.no_grad():
    z = torch.randn(1, 512).to(device)
    latent_code = generator.model.mapping(z)  # Corrected line: Removed 'None'
latent_code.requires_grad_(True)

# Optimizer for latent code
optimizer = optim.Adam([latent_code], lr=0.01)

# Training process
num_iterations = 150000
layer_weights = [1, 1, 1, 1, 1, 1, 1, 1]  # Weights for VGG16 layers

# Modify easy_synthesize to work with torch tensors
def easy_synthesize(generator, latent_codes, latent_space_type='W', **kwargs):
    """Modified easy_synthesize function that works with PyTorch tensors."""
    # Preprocess latent codes
    latent_codes = generator.preprocess(latent_codes, latent_space_type=latent_space_type, **kwargs)

    # Synthesize images
    outputs = generator.synthesize(latent_codes, latent_space_type=latent_space_type, **kwargs)

    # Post-process images (convert from [-1, 1] to [0, 1])
    if 'image' in outputs:
        images = outputs['image']
        images = (images + 1) / 2.0  # Scale to [0, 1]
        outputs['image'] = images

    return outputs

# Ensure that the generator's preprocess and synthesize methods accept torch tensors and maintain the computation graph.

for iteration in range(num_iterations):
    optimizer.zero_grad()  # Clear previous gradients

    # Generate image using the modified easy_synthesize function
    synthesized = easy_synthesize(generator, latent_code, latent_space_type='W')
    generated_image = synthesized['image']

    # Ensure the generated image has 3 channels for VGG16
    if generated_image.shape[1] != 3:
        generated_image = generated_image[:, :3, :, :]  # Keep only the first 3 channels

    # Resize and normalize the generated image for VGG16
    generated_image_resized = F.interpolate(
        generated_image, size=(224, 224), mode='bilinear', align_corners=False
    )

    # Normalize using VGG16's preprocessing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    generated_image_resized = normalize(generated_image_resized.squeeze(0)).unsqueeze(0)

    # Extract features from the generated image using VGG16
    generated_features = vgg16_extractor(generated_image_resized)

    # Compute losses
    losses = []
    total_loss = 0
    for i in range(len(vgg16_layers)):
        loss = F.mse_loss(generated_features[i], target_features[i])
        weighted_loss = layer_weights[i] * loss
        total_loss += weighted_loss
        losses.append(loss.item())

    # Calculate intra-channel color diversity loss (optional)
    # Remove or comment out if not needed
    # total_loss += 2.5 * color_diversity_loss  # Adjust weight as needed

    total_loss.backward()
    optimizer.step()

    # Print progress
    if iteration % 100 == 0 or iteration == num_iterations - 1:
        loss_str = ', '.join([f"Layer {vgg16_layers[i]}: {losses[i]:.4f}" for i in range(len(losses))])
        print(f"Iteration {iteration}, Total Loss: {total_loss.item():.4f}, Losses: {loss_str}")

# Save the optimized latent vector
if os.path.exists("/data"):
    latent_save_path = "/data/optimized_ws1.npy"
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    latent_save_path = os.path.join(script_dir, "optimized_ws1.npy")

try:
    np.save(latent_save_path, latent_code.detach().cpu().numpy())
    print(f"Latent vector saved successfully to: {latent_save_path}")
except Exception as e:
    print(f"Error saving latent vector: {e}")

# Generate the final image using the optimized latent code
synthesized = easy_synthesize(generator, latent_code, latent_space_type='W')
generated_image = synthesized['image'].detach().cpu()

# Move the image to CPU and convert to NumPy array
final_generated_image_np = generated_image.squeeze(0).permute(1, 2, 0).numpy()

# Clip values to ensure they are in [0, 1]
final_generated_image_np = np.clip(final_generated_image_np, 0, 1)

# Save the final generated image
final_image_path = os.path.join(script_dir, "image215.png")
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
