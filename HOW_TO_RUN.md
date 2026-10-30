# How to Run and Use the ABSA Project

A practical, step-by-step guide to run, test, and demonstrate the Aspect-Based Sentiment Analysis system on Windows 11.

---

## 1. Quick Start (Run Both Backend & Frontend)

### Terminal 1: Start the FastAPI Backend
Open PowerShell in the workspace:
```powershell
cd "c:\Users\sanga\Desktop\BOOKS\SEMESTER 5\NLP\NLP Project\absa"
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
You will see:
```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```
- **API Status:** `http://127.0.0.1:8000/health`
- **Swagger Documentation:** `http://127.0.0.1:8000/docs`

---

### Terminal 2: Start the Stripe-Style Web Dashboard
Open a **second** PowerShell window:
```powershell
cd "c:\Users\sanga\Desktop\BOOKS\SEMESTER 5\NLP\NLP Project\absa\frontend"
npm run dev
```
You will see:
```text
  VITE v8.3.5  ready in 150 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```
- Open your browser at: **`http://localhost:5173`**

---

## 2. Using the Dashboard Features

### A. Single Review Analysis
1. Select the **Domain** (Restaurant or Laptop).
2. Select the **Model Architecture** (DistilBERT is recommended; BiLSTM is baseline).
3. Choose a quick-fill sample or paste your own review, for example:
   > *"The pizza was delicious and crispy, but the service was extremely slow."*
4. Click **Analyze Review**.
5. Observe:
   - Extracted aspects highlighted in the text.
   - Separate cards for each aspect (`pizza` -> Positive 96%, `service` -> Negative 94%).
   - Confidence progress bars.
   - **Influential Context Tokens** explaining why the model predicted that sentiment (e.g. `delicious` and `crispy` supporting positive polarity).
   - Export buttons (**Export JSON**, **Export CSV**).

### B. Batch Processing Tab
1. Click the **Batch Processing** tab at the top.
2. Enter multiple customer reviews (one per line).
3. Click **Process Batch**.
4. A structured table will display every review with all extracted aspects and polarities.

### C. Model Benchmarks Tab
1. Click the **Model Benchmarks** tab.
2. View the empirical test set metrics (Exact Match F1, Token F1, Sentiment Accuracy, Macro-F1) comparing DistilBERT against BiLSTM across both domains.

---

## 3. Alternative: Running the Streamlit Dashboard

If you prefer to run the Streamlit interface:
```powershell
cd "c:\Users\sanga\Desktop\BOOKS\SEMESTER 5\NLP\NLP Project\absa"
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```
It will automatically open in your default browser at `http://localhost:8501`.

---

## 4. Running the Automated Test Suite

To verify all components across the entire repository:
```powershell
cd "c:\Users\sanga\Desktop\BOOKS\SEMESTER 5\NLP\NLP Project\absa"
.\.venv\Scripts\pytest.exe -v
```
**Expected Output:**
```text
============================= 40 passed in 9.71s ==============================
```

---

## 5. Using the REST API via cURL or Python

### Example 1: Predict via cURL
```powershell
curl -X POST "http://127.0.0.1:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{\"review\": \"The battery life is exceptional and the display is bright, but the keyboard feels cheap.\", \"model\": \"distilbert\", \"domain\": \"laptop\", \"explain\": true}'
```

### Example 2: Predict via Python Script
```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "review": "The food was delicious, but the waiter was rude.",
    "model": "distilbert",
    "domain": "restaurant",
    "explain": True,
    "confidence_threshold": 0.3
}
response = requests.post(url, json=payload)
print(response.json())
```

---

## 6. How to Re-Run Evaluation & Re-Train (If Needed)

### Re-Run Evaluation:
```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate
```
This re-evaluates all 4 models on the test sets and saves plots to `results/plots/`.

### Re-Train Models:
```powershell
.\.venv\Scripts\python.exe -m src.training.train_all
```
This re-trains BiLSTM and DistilBERT extractors and classifiers on both domains.
