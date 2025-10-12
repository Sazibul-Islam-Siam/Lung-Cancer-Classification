# 🫁 Lung Cancer Classification Web Application

An AI-powered Flask web application for lung cancer screening and classification using deep learning. The app provides real-time diagnosis with Grad-CAM visualization to highlight regions of interest in medical images.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-orange)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3.2-purple)

## 📋 Features

- **🔬 AI-Powered Classification**: Classifies lung tissue into 3 categories:
  - Adenocarcinoma (Malignant)
  - Benign (Normal Tissue)
  - Squamous Cell Carcinoma (Malignant)

- **🔥 Grad-CAM Visualization**: Highlights the regions the AI model focused on for making predictions

- **📊 Confidence Scores**: Displays probability distribution for all classes

- **💻 Modern UI**: Professional Bootstrap 5 interface with Font Awesome icons

- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

- **⚡ Fast Processing**: Real-time inference with optimized model architecture

## 🏗️ Model Architecture

The application uses **MBv3toViT** - a hybrid architecture combining:
- **MobileNetV3-Small** backbone for efficient feature extraction
- **Vision Transformer (ViT)** encoder for advanced pattern recognition
- **Linear classification head** for 3-class output

**Model Specifications:**
- Input size: 224x224 RGB images
- ViT dimension: 768
- Transformer depth: 8 layers
- Attention heads: 12
- Output classes: 3

## 📁 Project Structure

```
Lung Cancer Classification/
│
├── app.py                 # Main Flask application
├── ok.py                  # Model architecture definition (MBv3toViT)
├── model.pth              # Pre-trained model weights
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
│
├── templates/
│   ├── index.html         # Upload interface
│   └── result.html        # Results dashboard
│
└── uploads/               # Temporary storage for uploaded images
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd "Lung Cancer Classification"

# Or download and extract the ZIP file
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installation may take 5-10 minutes depending on your internet speed, as PyTorch is a large package.

### Step 4: Verify Required Files

Ensure these files are present:
- ✅ `app.py` - Main application
- ✅ `ok.py` - Model architecture
- ✅ `model.pth` - Pre-trained model weights (required!)
- ✅ `templates/` folder with HTML files

### Step 5: Run the Application

```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Step 6: Open in Browser

Navigate to: **http://127.0.0.1:5000** or **http://localhost:5000**

## 📖 Usage Guide

### 1. Upload Image
- Click the upload area or drag and drop an image
- Supported formats: PNG, JPG, JPEG, DICOM
- Maximum file size: 16MB

### 2. Analyze
- Click the **"Analyze Scan"** button
- Wait for AI processing (typically 1-3 seconds)

### 3. View Results
The results page displays:
- **Original Image**: Your uploaded scan
- **Diagnosis Card**: Classification result with confidence score
- **Probability Distribution**: Bar chart showing all class probabilities
- **Grad-CAM Heatmap**: Visualization of model attention areas
- **Statistics**: Key metrics and analysis details

### 4. Next Steps
- Click **"Analyze Another Scan"** for new analysis
- Click **"Print Report"** to save/print the diagnostic report

## 🔧 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'X'**
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**2. Model file not found**
```
Error: [Errno 2] No such file or directory: 'model.pth'
```
**Solution:** Ensure `model.pth` is in the same directory as `app.py`

**3. CUDA/GPU Issues**
```
RuntimeError: Attempting to deserialize object on a CUDA device
```
**Solution:** The app automatically handles this. If issues persist, the code uses CPU by default.

**4. Port 5000 already in use**
```bash
# Solution: Change port in app.py (last line)
app.run(debug=True, port=5001)  # Use different port
```

**5. Image upload fails**
- Check file format (must be image file)
- Ensure file size < 16MB
- Try different image

## 🛠️ Configuration

### Change Upload Settings

Edit `app.py`:
```python
app.config['UPLOAD_FOLDER'] = 'uploads'  # Upload directory
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
```

### Modify Allowed Extensions

Edit `app.py`:
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
```

### Update Model Classes

If using different classes, edit `app.py`:
```python
class_names = ['class1', 'class2', 'class3']  # Update names
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 2.3.3 | Web framework |
| PyTorch | 2.0.1 | Deep learning |
| torchvision | 0.15.2 | Image transforms |
| timm | 0.9.12 | Vision Transformer |
| Pillow | 10.0.1 | Image processing |
| opencv-python | 4.8.1.78 | Grad-CAM visualization |
| matplotlib | 3.8.0 | Plotting |
| numpy | 1.24.3 | Numerical operations |

## ⚠️ Medical Disclaimer

**IMPORTANT:** This application is designed for educational and research purposes only.

- Results are for **reference only** and should **NOT** replace professional medical diagnosis
- All findings **MUST** be reviewed by a qualified radiologist or physician
- Do not use for clinical decision-making without proper medical validation
- This is a demonstration tool, not a certified medical device

## 🔒 Privacy & Security

- Uploaded images are temporarily stored in the `uploads/` folder
- Files are automatically cleaned up after processing
- No data is sent to external servers
- All processing happens locally on your machine

## 🎨 UI/UX Features

- **Bootstrap 5**: Modern, responsive design framework
- **Font Awesome**: Professional icon library
- **Inter Font**: Clean, readable typography
- **Gradient Backgrounds**: Visually appealing interface
- **Animated Progress Bars**: Smooth probability visualization
- **Drag & Drop**: Intuitive file upload
- **Real-time Preview**: See uploaded image before analysis

## 📊 Performance

- **Inference Time**: ~1-3 seconds per image (CPU)
- **Memory Usage**: ~2-4 GB RAM
- **Image Size**: Automatically resized to 224x224
- **Batch Processing**: Currently single image only

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Batch processing support
- Additional visualization options
- Export to PDF reports
- Database integration for history
- REST API endpoints

## 📄 License

This project is for educational purposes. Please ensure you have proper rights to use the model weights.

## 👨‍💻 Technical Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Ensure Python version is 3.8+
4. Check that `model.pth` file exists

## 🔄 Version History

- **v1.0.0** - Initial release with Bootstrap UI and Grad-CAM
  - 3-class classification (Adenocarcinoma, Benign, Squamous Cell Carcinoma)
  - Grad-CAM visualization
  - Modern responsive interface
  - Probability distribution charts

---

**Made with ❤️ for Medical AI Research**