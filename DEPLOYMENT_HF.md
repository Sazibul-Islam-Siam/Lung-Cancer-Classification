# 🚀 Deploy to Hugging Face Spaces (RECOMMENDED)

## Why Hugging Face Spaces?
✅ **16GB RAM** (vs Render's 512MB)  
✅ **Perfect for ML models**  
✅ **100% FREE**  
✅ **No credit card required**  
✅ **Git LFS supported natively**  
✅ **Auto-detects model files**

---

## 📋 Quick Deploy (5 Minutes)

### Step 1: Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Sign up (free)
3. Verify your email

### Step 2: Create a New Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `lung-cancer-classification`
   - **License**: `mit`
   - **Select SDK**: `Gradio`
   - **Space hardware**: `CPU basic` (free)
   - **Visibility**: `Public`
3. Click **"Create Space"**

### Step 3: Upload Files

#### Option A: Upload via Web (Easiest)
1. In your new Space, click **"Files"** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload these files from your project:
   ```
   app_gradio.py       (rename to: app.py)
   model.pth
   requirements_hf.txt (rename to: requirements.txt)
   README_HF.md        (rename to: README.md)
   ```
4. Click **"Commit changes to main"**

#### Option B: Push via Git (Advanced)
```bash
# In your project folder
git clone https://huggingface.co/spaces/YOUR_USERNAME/lung-cancer-classification
cd lung-cancer-classification

# Copy files
cp app_gradio.py app.py
cp requirements_hf.txt requirements.txt  
cp README_HF.md README.md
cp model.pth .

# Push
git add .
git commit -m "Initial commit"
git push
```

### Step 4: Wait for Build
- Build takes ~5-8 minutes
- Watch the **"Logs"** tab
- When done, you'll see **"Running"** status

### Step 5: Your App is Live! 🎉
- URL: `https://huggingface.co/spaces/YOUR_USERNAME/lung-cancer-classification`
- Share this link with anyone!

---

## 🎯 What You Get

Your app will have:
- ✅ Clean Gradio interface
- ✅ File upload
- ✅ Real-time predictions
- ✅ Probability charts
- ✅ Medical disclaimer
- ✅ Fast inference (16GB RAM)

---

## 🔧 Customize Your Space

After deployment, you can:
1. Click **"Edit"** → **"Files and versions"**
2. Edit `app.py` directly in browser
3. Changes auto-deploy

---

## 📊 Monitor Usage

- Go to your Space
- Click **"Analytics"** tab
- See visitor stats, usage, etc.

---

## 💡 Tips

**Model file too large?**
- Hugging Face auto-handles Git LFS
- Your model.pth (224MB) uploads fine

**Want GPU?**
- Click **"Settings"** → **"Change hardware"**
- Upgrade to GPU (paid, but fast)

**Share with team?**
- Add collaborators in Settings
- They can edit and deploy

---

## 🆚 Comparison

| Feature | Hugging Face | Render Free |
|---------|-------------|-------------|
| RAM | 16GB | 512MB ❌ |
| Build time | 5-8 min | 12-15 min |
| ML-optimized | Yes ✅ | No |
| Cost | Free | Free |
| GPU option | Yes | No |

---

## ✅ Next Steps

1. Follow Step 1-5 above
2. Wait 5-8 minutes
3. Get your link
4. Share with the world!

**Your app will be at:**
`https://huggingface.co/spaces/YOUR_USERNAME/lung-cancer-classification`

---

Need help? The Hugging Face community is super helpful: https://discuss.huggingface.co/
