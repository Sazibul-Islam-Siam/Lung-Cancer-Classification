How to add screenshots for the project

Place your screenshot files in this directory with the following recommended filenames:

- upload-page.png  (recommended 1024x1024 or similar square image)
- result-page.png  (recommended 2048x768 or similar landscape image)

Naming and size recommendations help keep the README and docs consistent. To add screenshots to the repo, run:

```powershell
# from project root
mkdir -p assets\screenshots
# copy your images into that folder (Windows example)
copy C:\path\to\upload-page.png assets\screenshots\upload-page.png
copy C:\path\to\result-page.png assets\screenshots\result-page.png

git add assets/screenshots/upload-page.png assets/screenshots/result-page.png assets/screenshots/README.md
git commit -m "docs: add screenshots"
git push origin main
```

If you don't want to store binary images in the repository, consider hosting them externally (GitHub Release, public S3/GCS, or an image CDN) and referencing the URLs in the README.
