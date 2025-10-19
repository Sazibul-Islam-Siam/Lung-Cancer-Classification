# Deployment Guide - Render.com

This guide will help you deploy your Lung Cancer Classification app to Render.com for free.

## Prerequisites
- GitHub account
- Render.com account (free - sign up at https://render.com)

## Step 1: Push Code to GitHub
Your code is already on GitHub at: https://github.com/Sazibul-Islam-Siam/Lung-Cancer-Classification

## Step 2: Deploy on Render

### Option A: Using Render Dashboard (Recommended)

1. **Go to Render Dashboard**
   - Visit https://render.com and sign in
   - Click "New +" → "Web Service"

2. **Connect GitHub Repository**
   - Select "Build and deploy from a Git repository"
   - Click "Connect" next to GitHub
   - Find and select your repository: `Lung-Cancer-Classification`

3. **Configure the Web Service**
   Fill in these settings:
   
   - **Name**: `lung-cancer-classification` (or any name you prefer)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Runtime**: `Docker`
   - **Instance Type**: `Free` (or upgrade if needed)

4. **Environment Variables** (Optional but recommended)
   Click "Advanced" → Add environment variable:
   - Key: `FLASK_DEBUG`
   - Value: `0`

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically:
     - Build your Docker image
     - Install dependencies
     - Start your app
   - Build takes ~10-15 minutes for first deployment

6. **Access Your App**
   - Once deployed, you'll get a URL like: `https://lung-cancer-classification-xxxx.onrender.com`
   - Share this link with anyone!

### Important Notes

⚠️ **Model File Size**
Your `model.pth` file (224 MB) is tracked with Git LFS. Render should pull it automatically. If you encounter issues:
- The model file is included in your repo via Git LFS
- Render's free tier supports this

⚠️ **Free Tier Limitations**
- App spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds to wake up
- 750 hours/month free (enough for continuous use)

⚠️ **Build Time**
- First build: ~10-15 minutes (installing PyTorch)
- Subsequent builds: ~5-8 minutes (if dependencies unchanged)

### Troubleshooting

**Build fails during Docker build:**
- Check build logs in Render dashboard
- Common issue: memory limits during pip install
- Solution: Dockerfile already uses CPU-only PyTorch wheels to reduce build size

**App crashes on startup:**
- Check application logs in Render dashboard
- Verify `model.pth` was pulled correctly
- Check for import errors

**Slow first load:**
- This is normal on free tier (cold start)
- Upgrade to paid tier for always-on instances

## Alternative Deployment Options

### Option B: Railway.app
1. Visit https://railway.app
2. Connect GitHub repo
3. Deploy (similar process to Render)
4. Free tier: $5 credit/month

### Option C: Fly.io
1. Install flyctl: `https://fly.io/docs/hands-on/install-flyctl/`
2. Run: `fly launch`
3. Follow prompts
4. Free tier: 3GB persistent storage

## Monitoring Your Deployment

After deployment on Render:
- **Logs**: Dashboard → Your Service → Logs tab
- **Metrics**: Dashboard → Your Service → Metrics tab
- **Restart**: Dashboard → Manual Deploy → "Clear build cache & deploy"

## Updating Your App

When you make changes:
```bash
git add .
git commit -m "your changes"
git push origin main
```

Render will automatically:
- Detect the push
- Rebuild your app
- Deploy new version

---

**Your app will be live at**: `https://your-app-name.onrender.com`

Share this link with anyone and they can use your lung cancer classification tool!
