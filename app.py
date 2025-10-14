from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import mobilenet_v3_small
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from torch.nn import functional as F
import traceback

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
class MBv3toViT(nn.Module):
    def __init__(self, num_classes, vit_dim=768, vit_depth=8, vit_heads=12, drop=0.1):
        super().__init__()
        # 1) MobileNetV3 backbone (feature extractor)
        mb = mobilenet_v3_small(pretrained=True)
        self.backbone = mb.features  # outputs ~[B, 576, 7, 7]

        # Channel dimension coming out of MBv3-small:
        self.cnn_out_ch = 576  # (check via a dummy forward if you change MBv3 config)

        # 2) Project CNN channels -> ViT embed dim
        self.proj = nn.Conv2d(self.cnn_out_ch, vit_dim, kernel_size=1)

        # 3) ViT encoder (no patch embed, we already have tokens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 49, vit_dim))  # 49 for 7×7
        self.pos_drop = nn.Dropout(drop)

        # a compact ViT encoder from timm (set patch_size=1; we won’t use its patch embed)
        self.vit = VisionTransformer(
            img_size=14, patch_size=1, in_chans=vit_dim, num_classes=num_classes,
            embed_dim=vit_dim, depth=vit_depth, num_heads=vit_heads, mlp_ratio=4.0,
            qkv_bias=True, drop_rate=drop, attn_drop_rate=drop, drop_path_rate=0.0
        )
        # We’ll bypass ViT’s own patch embedding and class token handling.

        # Replace its head only; we’ll drive the encoder blocks directly
        self.head = nn.Linear(vit_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        # CNN features: [B, 576, 7, 7]
        f = self.backbone(x)
        # Project to ViT dim: [B, D, 7, 7]
        f = self.proj(f)
        # Flatten to tokens: [B, 49, D]
        f = f.flatten(2).transpose(1, 2)

        # Add CLS + positions
        cls = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
        tokens = torch.cat([cls, f], dim=1)          # [B, 50, D]
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # Feed through ViT blocks (bypass patch embed; use blocks + norm)
        for blk in self.vit.blocks:
            tokens = blk(tokens)
        tokens = self.vit.norm(tokens)
        cls_out = tokens[:, 0]                       # [B, D]

        return self.head(cls_out)

# Set up the path to your model and load it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MBv3toViT(num_classes=3)  # Number of classes in your dataset
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Define transformations for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Class names (you should update these based on your dataset)
class_names = ["lung_aca", "lung_n", "lung_scc"]  # Order from model training

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_loss = model_output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class

def generate_gradcam_heatmap(model, img_tensor, original_img_path, predicted_class):
    """Generate and save Grad-CAM heatmap"""
    try:
        # Get the last convolutional layer of the backbone
        target_layer = model.backbone[-1]  # Last layer of MobileNetV3 backbone
        
        # Create Grad-CAM object
        grad_cam = GradCAM(model, target_layer)
        
        # Generate CAM
        cam, _ = grad_cam.generate_cam(img_tensor, target_class=predicted_class)
        
        # Load original image
        original_img = cv2.imread(original_img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (224, 224))
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on original image
        superimposed = heatmap * 0.4 + original_img * 0.6
        superimposed = np.uint8(superimposed)
        
        # Save the Grad-CAM image
        gradcam_filename = 'gradcam_' + os.path.basename(original_img_path)
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Grad-CAM overlay
        plt.subplot(1, 2, 2)
        plt.imshow(superimposed)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(gradcam_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return gradcam_filename
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Server-side validation: quick heuristic to check for stained histopathology images
        try:
            pil_img = Image.open(file_path).convert('RGB')
        except Exception as e:
            return render_template('index.html', error="Invalid image file. Please upload a valid image.")

        try:
            # Convert to numpy and compute mean saturation in HSV color space
            import cv2
            arr = np.array(pil_img)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            sat_mean = float(hsv[:, :, 1].mean()) / 255.0
            # Heuristic thresholds: stained histology images tend to have noticeable saturation
            if sat_mean < 0.12:
                return render_template('index.html', error="Invalid image: not a stained histopathology image. Please upload an H&E-stained slide or similar.")
        except Exception:
            # If validation fails for any reason, continue but guard inference in try/except below
            pass

        # Open the image, apply transformations
        try:
            img = transform(pil_img).unsqueeze(0).to(device)  # Add batch dimension and move to device

            # Make prediction
            with torch.no_grad():
                outputs = model(img)
                probs = torch.softmax(outputs, dim=1)  # Get class probabilities
                all_probs = probs[0].cpu().numpy()
                confidences, predicted_class = torch.max(probs, 1)  # Get highest probability class

            predicted_class = predicted_class.item()
            confidence = confidences.item() * 100

            # Generate Grad-CAM heatmap
            gradcam_filename = generate_gradcam_heatmap(model, img, file_path, predicted_class)
        except Exception as e:
            tb = traceback.format_exc()
            # Log the full traceback to console for debugging
            print("Inference error:\n", tb)
            # Return a user-friendly error message
            return render_template('index.html', error="Unable to process this image. Please upload a stained histopathology image (H&E) or check image format.")

        # Map class names to full descriptions
        class_descriptions = {
            "lung_aca": "Adenocarcinoma",
            "lung_n": "Benign",
            "lung_scc": "Squamous Cell Carcinoma"
        }
        
        # Prepare all class probabilities
        all_class_probs = [
            {
                'name': class_descriptions.get(class_names[i], class_names[i]),
                'probability': float(all_probs[i] * 100)
            }
            for i in range(len(class_names))
        ]

        return render_template('result.html',
                             predicted_class=class_descriptions.get(class_names[predicted_class], class_names[predicted_class]),
                             confidence=confidence,
                             all_class_probs=all_class_probs,
                             image_filename=filename,
                             gradcam_filename=gradcam_filename)

if __name__ == '__main__':
    app.run(debug=True)
