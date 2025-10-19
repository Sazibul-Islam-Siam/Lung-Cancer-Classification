import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torchvision.models import mobilenet_v3_small
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

# Load the trained model
class MBv3toViT(nn.Module):
    def __init__(self, num_classes, vit_dim=768, vit_depth=8, vit_heads=12, drop=0.1):
        super().__init__()
        mb = mobilenet_v3_small(pretrained=True)
        self.backbone = mb.features
        self.cnn_out_ch = 576
        self.proj = nn.Conv2d(self.cnn_out_ch, vit_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 49, vit_dim))
        self.pos_drop = nn.Dropout(drop)
        self.vit = VisionTransformer(
            img_size=14, patch_size=1, in_chans=vit_dim, num_classes=num_classes,
            embed_dim=vit_dim, depth=vit_depth, num_heads=vit_heads, mlp_ratio=4.0,
            qkv_bias=True, drop_rate=drop, attn_drop_rate=drop, drop_path_rate=0.0
        )
        self.head = nn.Linear(vit_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        f = self.backbone(x)
        f = self.proj(f)
        f = f.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, f], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        for blk in self.vit.blocks:
            tokens = blk(tokens)
        tokens = self.vit.norm(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MBv3toViT(num_classes=3)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = ["Adenocarcinoma", "Benign", "Squamous Cell Carcinoma"]

def predict_image(image):
    if image is None:
        return None, None, "Please upload an image"
    
    # Validate image (saturation check)
    try:
        arr = np.array(image)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        sat_mean = float(hsv[:, :, 1].mean()) / 255.0
        if sat_mean < 0.12:
            return None, None, "⚠️ Invalid image: not a stained histopathology image. Please upload an H&E-stained slide."
    except:
        pass
    
    # Predict
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs = probs[0].cpu().numpy()
            predicted_class = torch.argmax(probs, 1).item()
            confidence = probs[0][predicted_class].item() * 100
        
        # Create probability chart
        prob_dict = {class_names[i]: float(all_probs[i] * 100) for i in range(len(class_names))}
        
        # Generate Grad-CAM
        target_layer = model.backbone[-1]
        model.zero_grad()
        outputs[0, predicted_class].backward()
        
        # Get gradients and activations
        gradients = target_layer.weight.grad if hasattr(target_layer, 'weight') else None
        
        result_text = f"""
## 🔬 Diagnosis Result

**Predicted Class:** {class_names[predicted_class]}
**Confidence:** {confidence:.2f}%

### 📊 Probability Distribution:
- Adenocarcinoma: {all_probs[0]*100:.2f}%
- Benign: {all_probs[1]*100:.2f}%
- Squamous Cell Carcinoma: {all_probs[2]*100:.2f}%

⚠️ **Medical Disclaimer:** This is for educational purposes only. Results should be reviewed by a qualified medical professional.
"""
        
        return image, prob_dict, result_text
        
    except Exception as e:
        return None, None, f"❌ Error processing image: {str(e)}"

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Lung Cancer Classification") as demo:
    gr.Markdown("""
    # 🫁 Lung Cancer Screening Tool
    ### AI-Powered Diagnostic Analysis for Histopathology Images
    
    Upload an H&E-stained histopathology image to classify lung tissue.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Histopathology Image")
            classify_btn = gr.Button("🔬 Analyze Scan", variant="primary", size="lg")
            
        with gr.Column():
            output_image = gr.Image(label="Uploaded Image")
            prob_plot = gr.BarPlot(
                x="class",
                y="probability",
                title="Classification Probabilities",
                y_lim=[0, 100],
                width=400,
                height=300
            )
            result_text = gr.Markdown()
    
    gr.Markdown("""
    ### ℹ️ Supported Classes:
    - **Adenocarcinoma** (Malignant)
    - **Benign** (Normal Tissue)  
    - **Squamous Cell Carcinoma** (Malignant)
    
    ### ⚡ Features:
    - Fast AI-powered analysis
    - Confidence scores for all classes
    - Optimized for H&E-stained slides
    """)
    
    classify_btn.click(
        fn=predict_image,
        inputs=[input_image],
        outputs=[output_image, prob_plot, result_text]
    )

if __name__ == "__main__":
    demo.launch()
