# Deployment Guide — Aspectra ABSA System

This guide outlines **3 simple ways** to deploy and share this project, ranging from an instant free public link to cloud hosting.

---

## Method 1: Unified Single-Command Local Deployment (Easiest)

You do **NOT** need two separate terminals if you just want to run the full application. The backend (`api/main.py`) is already configured to automatically serve the pre-compiled **Aspectra** frontend directly!

### Step 1: Open PowerShell and run:
```powershell
cd "c:\Users\sanga\Desktop\BOOKS\SEMESTER 5\NLP\NLP Project\absa"
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Step 2: Open Browser
- **Application URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000) (Serves the complete Stripe-style UI)
- **API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Interactive Swagger UI)

---

## Method 2: Instant Free Public Link (Best to Show Your Teacher)

If you want your teacher or classmates to open the application on their own phone, tablet, or laptop during your presentation, you can create a secure public HTTPS link in **10 seconds** using a free tunnel:

### Option A: Using `localtunnel` (No account needed)
1. In Terminal 1, keep the unified server running:
   ```powershell
   uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. In Terminal 2, run:
   ```powershell
   npx localtunnel --port 8000
   ```
3. It will give you a public URL (e.g. `https://friendly-tiger-42.loca.lt`).
4. Anyone who opens that link will see your live **Aspectra** application running directly off your GPU!

### Option B: Using `cloudflared` (Cloudflare Tunnel - 100% Free & Fast)
```powershell
# Run Cloudflare's free temporary tunnel
cloudflared tunnel --url http://127.0.0.1:8000
```
It gives an official `*.trycloudflare.com` HTTPS link with SSL certificate included.

---

## Method 3: Permanent Cloud Deployment (Render / Hugging Face Spaces)

If you want to host it permanently in the cloud 24/7 for free:

### Option A: Hugging Face Spaces (Recommended for NLP Projects)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **FastAPI** or **Docker** (free CPU or GPU hardware with 16GB RAM).
3. Upload the repository (`api/`, `frontend/dist/`, `models/`, `src/`, `requirements.txt`).
4. Set the startup command to:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 7860
   ```
5. Hugging Face gives you a permanent link: `https://huggingface.co/spaces/your-username/aspectra`.

### Option B: Render.com (Free Web Service)
1. Create a free account on [render.com](https://render.com).
2. Connect your GitHub repository.
3. Choose **Web Service** with Python environment.
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Render will assign you a live link: `https://aspectra.onrender.com`.

---

## Summary Checklist Before Showing Your Teacher:
- [x] Backend running on port 8000 (`uvicorn api.main:app`)
- [x] Frontend compiled in `frontend/dist` (`npx vite build`)
- [x] 40/40 tests passing (`pytest -v`)
- [x] Both Restaurant and Laptop domains tested
- [x] DistilBERT and BiLSTM models verified
- [x] Leave-one-out token explainability visible on review analysis
