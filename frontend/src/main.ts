import './style.css';

const API_BASE = 'http://127.0.0.1:8000';

interface Aspect {
  aspect: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  char_start?: number;
  char_end?: number;
  explanation?: {
    predicted_sentiment: string;
    confidence: number;
    top_influential_words: { word: string; importance: number }[];
    supporting_words: string[];
    opposing_words: string[];
  };
}

interface PredictionResponse {
  review: string;
  model: string;
  domain: string;
  aspects: Aspect[];
  latency_ms: number;
}

const SAMPLE_REVIEWS = {
  restaurant: [
    { label: "Mixed Sentiment", text: "The pizza was delicious and crispy, but the service was extremely slow." },
    { label: "Service vs Food", text: "The staff was super friendly and welcoming, though the pasta was rather bland." },
    { label: "Multi-Aspect Praise", text: "Outstanding wine selection, mouthwatering desserts, and a delightfully cozy ambiance!" },
  ],
  laptop: [
    { label: "Hardware Contrast", text: "The battery life is exceptional and the display is bright, but the keyboard feels cheap." },
    { label: "Performance Praise", text: "Lightweight chassis with blazing performance, gorgeous 4K display, and instantaneous boot times." },
    { label: "Thermal Issue", text: "The aluminum body overheats under heavy load and the cooling fan noise is unbearable." },
  ]
};

// Application State
let currentTab = 'single';
let currentDomain = 'restaurant';
let currentModel = 'distilbert';
let threshold = 0.3;
let explain = true;
let lastResult: PredictionResponse | null = null;
let isAnalyzing = false;

function renderApp() {
  const app = document.getElementById('app')!;
  app.innerHTML = `
    <div class="page-background"></div>
    
    <!-- Stripe Navigation -->
    <nav class="navbar">
      <div class="brand">
        <div class="brand-icon">A</div>
        <span>Aspectra</span>
      </div>
      <div class="system-status" id="system-status-badge">
        <span class="status-dot" id="system-status-dot"></span>
        <span id="system-status-text">Connecting to API...</span>
      </div>
    </nav>

    <main class="container">
      <!-- Hero -->
      <section class="hero">
        <span class="hero-pill">Enterprise NLP • Customer Review Intelligence</span>
        <h1>Aspect-Based Sentiment Analysis</h1>
        <p>
          Extract entity aspect terms and predict granular contextual polarities using 
          BiLSTM sequence models and fine-tuned DistilBERT transformers with token explainability.
        </p>

        <!-- KPI Summary Cards -->
        <div class="kpi-row">
          <div class="kpi-card emerald">
            <div class="kpi-label">Aspect Extraction F1</div>
            <div class="kpi-value" style="color: #059669;">86.07%</div>
            <div class="kpi-sub">DistilBERT Exact Match</div>
          </div>
          <div class="kpi-card blurple">
            <div class="kpi-label">Sentiment Accuracy</div>
            <div class="kpi-value" style="color: #4f46e5;">84.29%</div>
            <div class="kpi-sub">Fine-tuned on SemEval</div>
          </div>
          <div class="kpi-card cyan">
            <div class="kpi-label">Inference Latency</div>
            <div class="kpi-value" style="color: #0891b2;">~45 ms</div>
            <div class="kpi-sub">GPU Accelerated (RTX 2050)</div>
          </div>
          <div class="kpi-card amber">
            <div class="kpi-label">Gold Dataset</div>
            <div class="kpi-value" style="color: #d97706;">7,694</div>
            <div class="kpi-sub">100% Verified Offsets</div>
          </div>
        </div>
      </section>

      <!-- Segmented Tab Group -->
      <div class="tab-bar">
        <div class="tab-group">
          <button class="tab-btn ${currentTab === 'single' ? 'active' : ''}" data-tab="single">Live Review Analyzer</button>
          <button class="tab-btn ${currentTab === 'batch' ? 'active' : ''}" data-tab="batch">Batch Processing</button>
          <button class="tab-btn ${currentTab === 'metrics' ? 'active' : ''}" data-tab="metrics">Evaluation Benchmarks</button>
        </div>
      </div>

      <div id="tab-content"></div>
    </main>
  `;

  attachTabEvents();
  renderTabContent();
  checkApiHealth();
}

function attachTabEvents() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      currentTab = (e.currentTarget as HTMLElement).dataset.tab!;
      renderApp();
    });
  });
}

async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      const dot = document.getElementById('system-status-dot');
      const text = document.getElementById('system-status-text');
      if (dot && text) {
        dot.className = 'status-dot';
        text.textContent = `Operational • ${data.device.toUpperCase()}`;
      }
    }
  } catch {
    const dot = document.getElementById('system-status-dot');
    const text = document.getElementById('system-status-text');
    if (dot && text) {
      dot.className = 'status-dot offline';
      text.textContent = 'API Offline (Port 8000)';
    }
  }
}

function renderTabContent() {
  const container = document.getElementById('tab-content')!;
  if (currentTab === 'single') {
    renderSingleTab(container);
  } else if (currentTab === 'batch') {
    renderBatchTab(container);
  } else {
    renderMetricsTab(container);
  }
}

function renderSingleTab(container: HTMLElement) {
  const samples = SAMPLE_REVIEWS[currentDomain as keyof typeof SAMPLE_REVIEWS];
  const chipClasses = ['c-blue', 'c-purple', 'c-amber'];

  container.innerHTML = `
    <div class="dashboard-grid">
      <!-- Input Card -->
      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">Review Input & Configuration</h2>
            <div class="card-subtitle">Choose domain, model architecture, and input review</div>
          </div>
        </div>

        <div class="form-group">
          <label class="form-label">Review Domain</label>
          <select id="domain-select" class="form-select">
            <option value="restaurant" ${currentDomain === 'restaurant' ? 'selected' : ''}>Restaurant Reviews (Dining, Food, Service)</option>
            <option value="laptop" ${currentDomain === 'laptop' ? 'selected' : ''}>Laptop Reviews (Hardware, Screen, Battery)</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">Model Architecture</label>
          <select id="model-select" class="form-select">
            <option value="distilbert" ${currentModel === 'distilbert' ? 'selected' : ''}>DistilBERT (Fine-Tuned Transformer — Recommended)</option>
            <option value="bilstm" ${currentModel === 'bilstm' ? 'selected' : ''}>BiLSTM (Bidirectional LSTM Baseline)</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">Review Text</label>
          <textarea id="review-input" class="form-textarea" placeholder="Enter customer feedback...">${samples[0].text}</textarea>
          <div class="samples-wrapper">
            <span style="font-size: 0.78rem; color: var(--text-muted); font-weight: 700; margin-right: 2px;">Try:</span>
            ${samples.map((s, idx) => `
              <button class="sample-chip ${chipClasses[idx % 3]}" data-idx="${idx}">${s.label}</button>
            `).join('')}
          </div>
        </div>

        <div class="form-group">
          <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
            <label class="form-label" style="margin: 0;">Confidence Threshold</label>
            <span class="range-val" id="thresh-val">${Math.round(threshold * 100)}%</span>
          </div>
          <div class="slider-row">
            <input type="range" id="thresh-slider" min="0.0" max="1.0" step="0.05" value="${threshold}">
          </div>
        </div>

        <div class="form-group">
          <label class="checkbox-container">
            <input type="checkbox" id="explain-check" ${explain ? 'checked' : ''}>
            <span>Enable Token Attribution (Leave-one-out explainability)</span>
          </label>
        </div>

        <button id="analyze-btn" class="btn-primary" ${isAnalyzing ? 'disabled' : ''}>
          ${isAnalyzing ? 'Processing Inference...' : 'Analyze Review'}
        </button>
      </div>

      <!-- Results Card -->
      <div class="card" id="results-card">
        <div class="card-header">
          <div>
            <h2 class="card-title">Extracted Aspects & Sentiments</h2>
            <div class="card-subtitle">Granular polarities, token spans, and feature attributions</div>
          </div>
          <span id="latency-indicator" style="font-size: 0.8rem; font-weight: 700; color: var(--stripe-blurple);"></span>
        </div>

        <div id="results-viewport">
          <div style="text-align: center; padding: 3.5rem 1rem; color: var(--text-muted);">
            Click <strong>Analyze Review</strong> to run aspect term extraction and sentiment classification.
          </div>
        </div>
      </div>
    </div>
  `;

  // Attach Input Handlers
  const domainSelect = document.getElementById('domain-select') as HTMLSelectElement;
  domainSelect.addEventListener('change', () => {
    currentDomain = domainSelect.value;
    const reviewInput = document.getElementById('review-input') as HTMLTextAreaElement;
    reviewInput.value = SAMPLE_REVIEWS[currentDomain as keyof typeof SAMPLE_REVIEWS][0].text;
    renderSingleTab(container);
  });

  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  modelSelect.addEventListener('change', () => {
    currentModel = modelSelect.value;
  });

  const threshSlider = document.getElementById('thresh-slider') as HTMLInputElement;
  threshSlider.addEventListener('input', () => {
    threshold = parseFloat(threshSlider.value);
    document.getElementById('thresh-val')!.textContent = `${Math.round(threshold * 100)}%`;
  });

  const explainCheck = document.getElementById('explain-check') as HTMLInputElement;
  explainCheck.addEventListener('change', () => {
    explain = explainCheck.checked;
  });

  document.querySelectorAll('.sample-chip').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt((e.currentTarget as HTMLElement).dataset.idx!);
      const reviewInput = document.getElementById('review-input') as HTMLTextAreaElement;
      reviewInput.value = SAMPLE_REVIEWS[currentDomain as keyof typeof SAMPLE_REVIEWS][idx].text;
    });
  });

  document.getElementById('analyze-btn')!.addEventListener('click', handleAnalyze);

  if (lastResult) {
    displaySingleResults(lastResult);
  }
}

async function handleAnalyze() {
  const reviewInput = document.getElementById('review-input') as HTMLTextAreaElement;
  const text = reviewInput.value.trim();
  if (!text) return;

  isAnalyzing = true;
  const btn = document.getElementById('analyze-btn') as HTMLButtonElement;
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Processing Inference...';
  }

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        review: text,
        model: currentModel,
        domain: currentDomain,
        explain: explain,
        confidence_threshold: threshold
      })
    });

    if (!res.ok) throw new Error('API request failed');
    const data: PredictionResponse = await res.json();
    lastResult = data;
    displaySingleResults(data);
  } catch (err) {
    const viewport = document.getElementById('results-viewport')!;
    viewport.innerHTML = `
      <div style="background: var(--status-neg-bg); border: 1px solid var(--status-neg-border); padding: 1.25rem; border-radius: 8px; color: var(--status-neg-text);">
        <strong>API Disconnected</strong><br>
        Ensure the backend server is running in PowerShell via: <code>uvicorn api.main:app --reload</code>
      </div>
    `;
  } finally {
    isAnalyzing = false;
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Analyze Review';
    }
  }
}

function buildHighlightedText(reviewText: string, aspects: Aspect[]): string {
  interface Span { start: number; end: number; sentiment: string }
  const validSpans: Span[] = [];

  for (const a of aspects) {
    let s = a.char_start;
    let e = a.char_end;

    // Fallback search if index is missing or out of bounds
    if (s === undefined || e === undefined || s < 0 || e <= s || e > reviewText.length) {
      const idx = reviewText.toLowerCase().indexOf(a.aspect.toLowerCase());
      if (idx >= 0) {
        s = idx;
        e = idx + a.aspect.length;
      }
    }

    if (s !== undefined && e !== undefined && s >= 0 && e <= reviewText.length && e > s) {
      // Disallow overlapping spans to prevent word splitting
      const hasOverlap = validSpans.some(v => Math.max(s!, v.start) < Math.min(e!, v.end));
      if (!hasOverlap) {
        validSpans.push({ start: s, end: e, sentiment: a.sentiment });
      }
    }
  }

  // Sort strictly in descending order of start index
  validSpans.sort((a, b) => b.start - a.start);

  let html = reviewText;
  for (const span of validSpans) {
    const before = html.slice(0, span.start);
    const textSpan = html.slice(span.start, span.end);
    const after = html.slice(span.end);
    html = `${before}<span class="hl-${span.sentiment}">${textSpan}</span>${after}`;
  }
  return html;
}

function displaySingleResults(data: PredictionResponse) {
  const latencyIndicator = document.getElementById('latency-indicator');
  if (latencyIndicator) {
    latencyIndicator.textContent = `${data.latency_ms} ms • ${data.model.toUpperCase()}`;
  }

  const viewport = document.getElementById('results-viewport')!;
  if (!data.aspects || data.aspects.length === 0) {
    viewport.innerHTML = `
      <div style="text-align: center; padding: 3rem 1rem; color: var(--text-muted);">
        No aspects detected meeting the ${Math.round(threshold * 100)}% confidence threshold.
      </div>
    `;
    return;
  }

  // Build clean non-overlapping highlights
  const highlightedText = buildHighlightedText(data.review, data.aspects);

  let html = `
    <div class="highlight-panel">
      ${highlightedText}
    </div>
    <div class="aspects-list">
  `;

  for (const a of data.aspects) {
    html += `
      <div class="aspect-card ${a.sentiment}">
        <div class="aspect-top">
          <span class="aspect-title">${a.aspect}</span>
          <span class="badge-sentiment ${a.sentiment}">
            ● ${a.sentiment} (${Math.round(a.confidence * 100)}%)
          </span>
        </div>
        <div class="meter-bg">
          <div class="meter-fill ${a.sentiment}" style="width: ${Math.round(a.confidence * 100)}%"></div>
        </div>
    `;

    if (a.explanation && a.explanation.top_influential_words) {
      html += `
        <div class="explain-section">
          <span class="explain-label">Influential Tokens (Feature Attribution):</span>
          <div>
            ${a.explanation.top_influential_words.map(w => `
              <span class="tag-word ${w.importance >= 0 ? 'support' : 'oppose'}">
                ${w.word} (${w.importance > 0 ? '+' : ''}${w.importance.toFixed(2)})
              </span>
            `).join('')}
          </div>
        </div>
      `;
    }

    html += `</div>`;
  }

  html += `
    </div>
    <div class="export-bar">
      <button class="btn-secondary" id="export-json">Export JSON</button>
      <button class="btn-secondary" id="export-csv">Export CSV</button>
    </div>
  `;

  viewport.innerHTML = html;

  document.getElementById('export-json')?.addEventListener('click', () => {
    downloadFile(JSON.stringify(data, null, 2), 'absa_prediction.json', 'application/json');
  });

  document.getElementById('export-csv')?.addEventListener('click', () => {
    const csvRows = ['Aspect,Sentiment,Confidence,Review'];
    for (const a of data.aspects) {
      csvRows.push(`"${a.aspect}","${a.sentiment}",${a.confidence},"${data.review.replace(/"/g, '""')}"`);
    }
    downloadFile(csvRows.join('\n'), 'absa_prediction.csv', 'text/csv');
  });
}

function renderBatchTab(container: HTMLElement) {
  container.innerHTML = `
    <div class="card" style="max-width: 960px; margin: 0 auto;">
      <div class="card-header">
        <div>
          <h2 class="card-title">Batch Review Processing</h2>
          <div class="card-subtitle">Analyze entire cohorts of customer reviews concurrently</div>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem;">
        <div>
          <label class="form-label">Domain</label>
          <select id="batch-domain" class="form-select">
            <option value="restaurant">Restaurant Reviews</option>
            <option value="laptop">Laptop Reviews</option>
          </select>
        </div>
        <div>
          <label class="form-label">Model Architecture</label>
          <select id="batch-model" class="form-select">
            <option value="distilbert">DistilBERT (Transformer)</option>
            <option value="bilstm">BiLSTM (Baseline)</option>
          </select>
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">Customer Reviews (One review per line)</label>
        <textarea id="batch-textarea" class="form-textarea" style="min-height: 160px;">The pizza was awesome and crispy, but the drinks were overpriced.
Great screen resolution, snappy trackpad, and lightweight chassis.
The waitress was rude and the table was dirty.
Battery lasts for 10 hours without any heating issues.</textarea>
      </div>

      <button id="batch-submit-btn" class="btn-primary" style="max-width: 220px;">
        Process Batch
      </button>

      <div id="batch-results-viewport" style="margin-top: 1.75rem;"></div>
    </div>
  `;

  document.getElementById('batch-submit-btn')!.addEventListener('click', async () => {
    const textarea = document.getElementById('batch-textarea') as HTMLTextAreaElement;
    const lines = textarea.value.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) return;

    const btn = document.getElementById('batch-submit-btn') as HTMLButtonElement;
    btn.disabled = true;
    btn.textContent = 'Processing Batch...';

    const domain = (document.getElementById('batch-domain') as HTMLSelectElement).value;
    const model = (document.getElementById('batch-model') as HTMLSelectElement).value;

    try {
      const res = await fetch(`${API_BASE}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviews: lines,
          model: model,
          domain: domain,
          confidence_threshold: threshold
        })
      });

      const data = await res.json();
      const viewport = document.getElementById('batch-results-viewport')!;
      
      let html = `
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
          <h3 style="font-size: 1.05rem; font-weight: 700;">Results (${data.total_reviews} reviews analyzed in ${data.latency_ms} ms)</h3>
        </div>
        <table class="data-table">
          <thead>
            <tr>
              <th style="width: 40px;">#</th>
              <th>Customer Review</th>
              <th>Identified Aspects & Polarities</th>
            </tr>
          </thead>
          <tbody>
      `;

      data.results.forEach((r: PredictionResponse, i: number) => {
        html += `
          <tr>
            <td style="color: var(--text-muted); font-weight: 700;">${i + 1}</td>
            <td style="font-weight: 500;">${r.review}</td>
            <td>
              ${r.aspects.map(a => `
                <span class="badge-sentiment ${a.sentiment}" style="margin: 2px; display: inline-flex;">
                  ${a.aspect}: ${a.sentiment}
                </span>
              `).join('') || '<span style="color: var(--text-muted); font-size: 0.85rem;">No aspects detected</span>'}
            </td>
          </tr>
        `;
      });

      html += `</tbody></table>`;
      viewport.innerHTML = html;
    } catch {
      alert('Batch analysis request failed.');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Process Batch';
    }
  });
}

function renderMetricsTab(container: HTMLElement) {
  container.innerHTML = `
    <div class="card" style="max-width: 960px; margin: 0 auto;">
      <div class="card-header">
        <div>
          <h2 class="card-title">Empirical Model Benchmarks</h2>
          <div class="card-subtitle">Official evaluation on unseen SemEval benchmark test sets</div>
        </div>
      </div>

      <h3 style="font-size: 0.95rem; font-weight: 800; color: var(--stripe-blurple); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
        Restaurant Domain (610 test reviews, 1,120 aspects)
      </h3>
      <table class="data-table" style="margin-bottom: 2rem;">
        <thead>
          <tr>
            <th>Architecture</th>
            <th>Extraction (Token F1)</th>
            <th>Extraction (Exact F1)</th>
            <th>Sentiment Accuracy</th>
            <th>Sentiment Macro-F1</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>DistilBERT (Fine-Tuned)</strong></td>
            <td><strong style="color: #059669;">85.54%</strong></td>
            <td><strong style="color: #059669;">86.07%</strong></td>
            <td><strong style="color: #059669;">84.29%</strong></td>
            <td><strong style="color: #059669;">76.98%</strong></td>
          </tr>
          <tr>
            <td>BiLSTM (Baseline)</td>
            <td>76.83%</td>
            <td>77.62%</td>
            <td>68.48%</td>
            <td>47.71%</td>
          </tr>
          <tr style="background: #f8fafc; font-style: italic; color: var(--text-muted);">
            <td>Transformer Advantage</td>
            <td>+8.71%</td>
            <td>+8.45%</td>
            <td>+15.81%</td>
            <td>+29.27%</td>
          </tr>
        </tbody>
      </table>

      <h3 style="font-size: 0.95rem; font-weight: 800; color: var(--stripe-blurple); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
        Laptop Domain (418 test reviews, 638 aspects)
      </h3>
      <table class="data-table" style="margin-bottom: 2rem;">
        <thead>
          <tr>
            <th>Architecture</th>
            <th>Extraction (Token F1)</th>
            <th>Extraction (Exact F1)</th>
            <th>Sentiment Accuracy</th>
            <th>Sentiment Macro-F1</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>DistilBERT (Fine-Tuned)</strong></td>
            <td><strong style="color: #059669;">82.95%</strong></td>
            <td><strong style="color: #059669;">81.29%</strong></td>
            <td><strong style="color: #059669;">75.71%</strong></td>
            <td><strong style="color: #059669;">71.56%</strong></td>
          </tr>
          <tr>
            <td>BiLSTM (Baseline)</td>
            <td>70.80%</td>
            <td>67.00%</td>
            <td>60.66%</td>
            <td>49.68%</td>
          </tr>
          <tr style="background: #f8fafc; font-style: italic; color: var(--text-muted);">
            <td>Transformer Advantage</td>
            <td>+12.15%</td>
            <td>+14.29%</td>
            <td>+15.05%</td>
            <td>+21.88%</td>
          </tr>
        </tbody>
      </table>

      <div style="background: #f8fafc; border: 1px solid var(--border-subtle); border-radius: 10px; padding: 1.25rem;">
        <h4 style="font-size: 0.92rem; font-weight: 800; margin-bottom: 0.5rem; color: var(--text-primary);">Key Architectural Insights:</h4>
        <ul style="padding-left: 1.25rem; font-size: 0.88rem; color: var(--text-secondary); line-height: 1.75;">
          <li><strong>Cross-Attention Routing:</strong> DistilBERT resolves contrastive sentiments across conjunctions (e.g., *"great pizza but rude staff"*).</li>
          <li><strong>Multi-Token Span Preservation:</strong> Exact match boundaries for phrases like *"battery life"* and *"operating system"* improve by **+14.3%**.</li>
          <li><strong>Edge vs Cloud Tradeoff:</strong> BiLSTM offers ultra-fast ~15ms inference on CPU, whereas DistilBERT provides high-precision enterprise accuracy (~45ms on GPU).</li>
        </ul>
      </div>
    </div>
  `;
}

function downloadFile(content: string, filename: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Start App
renderApp();
