(function(){let e=document.createElement(`link`).relList;if(e&&e.supports&&e.supports(`modulepreload`))return;for(let e of document.querySelectorAll(`link[rel="modulepreload"]`))n(e);new MutationObserver(e=>{for(let t of e)if(t.type===`childList`)for(let e of t.addedNodes)e.tagName===`LINK`&&e.rel===`modulepreload`&&n(e)}).observe(document,{childList:!0,subtree:!0});function t(e){let t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),t.credentials=e.crossOrigin===`use-credentials`?`include`:e.crossOrigin===`anonymous`?`omit`:`same-origin`,t}function n(e){if(e.ep)return;e.ep=!0;let n=t(e);fetch(e.href,n)}})();var e=`http://127.0.0.1:8000`,t={restaurant:[{label:`Mixed Sentiment`,text:`The pizza was delicious and crispy, but the service was extremely slow.`},{label:`Service vs Food`,text:`The staff was super friendly and welcoming, though the pasta was rather bland.`},{label:`Multi-Aspect Praise`,text:`Outstanding wine selection, mouthwatering desserts, and a delightfully cozy ambiance!`}],laptop:[{label:`Hardware Contrast`,text:`The battery life is exceptional and the display is bright, but the keyboard feels cheap.`},{label:`Performance Praise`,text:`Lightweight chassis with blazing performance, gorgeous 4K display, and instantaneous boot times.`},{label:`Thermal Issue`,text:`The aluminum body overheats under heavy load and the cooling fan noise is unbearable.`}]},n=`single`,r=`restaurant`,i=`distilbert`,a=.3,o=!0,s=null,c=!1;function l(){let e=document.getElementById(`app`);e.innerHTML=`
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
          <button class="tab-btn ${n===`single`?`active`:``}" data-tab="single">Live Review Analyzer</button>
          <button class="tab-btn ${n===`batch`?`active`:``}" data-tab="batch">Batch Processing</button>
          <button class="tab-btn ${n===`metrics`?`active`:``}" data-tab="metrics">Evaluation Benchmarks</button>
        </div>
      </div>

      <div id="tab-content"></div>
    </main>
  `,u(),f(),d()}function u(){document.querySelectorAll(`.tab-btn`).forEach(e=>{e.addEventListener(`click`,e=>{n=e.currentTarget.dataset.tab,l()})})}async function d(){try{let t=await fetch(`${e}/health`);if(t.ok){let e=await t.json(),n=document.getElementById(`system-status-dot`),r=document.getElementById(`system-status-text`);n&&r&&(n.className=`status-dot`,r.textContent=`Operational • ${e.device.toUpperCase()}`)}}catch{let e=document.getElementById(`system-status-dot`),t=document.getElementById(`system-status-text`);e&&t&&(e.className=`status-dot offline`,t.textContent=`API Offline (Port 8000)`)}}function f(){let e=document.getElementById(`tab-content`);n===`single`?p(e):n===`batch`?_(e):v(e)}function p(e){let n=t[r],l=[`c-blue`,`c-purple`,`c-amber`];e.innerHTML=`
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
            <option value="restaurant" ${r===`restaurant`?`selected`:``}>Restaurant Reviews (Dining, Food, Service)</option>
            <option value="laptop" ${r===`laptop`?`selected`:``}>Laptop Reviews (Hardware, Screen, Battery)</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">Model Architecture</label>
          <select id="model-select" class="form-select">
            <option value="distilbert" ${i===`distilbert`?`selected`:``}>DistilBERT (Fine-Tuned Transformer — Recommended)</option>
            <option value="bilstm" ${i===`bilstm`?`selected`:``}>BiLSTM (Bidirectional LSTM Baseline)</option>
          </select>
        </div>

        <div class="form-group">
          <label class="form-label">Review Text</label>
          <textarea id="review-input" class="form-textarea" placeholder="Enter customer feedback...">${n[0].text}</textarea>
          <div class="samples-wrapper">
            <span style="font-size: 0.78rem; color: var(--text-muted); font-weight: 700; margin-right: 2px;">Try:</span>
            ${n.map((e,t)=>`
              <button class="sample-chip ${l[t%3]}" data-idx="${t}">${e.label}</button>
            `).join(``)}
          </div>
        </div>

        <div class="form-group">
          <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
            <label class="form-label" style="margin: 0;">Confidence Threshold</label>
            <span class="range-val" id="thresh-val">${Math.round(a*100)}%</span>
          </div>
          <div class="slider-row">
            <input type="range" id="thresh-slider" min="0.0" max="1.0" step="0.05" value="${a}">
          </div>
        </div>

        <div class="form-group">
          <label class="checkbox-container">
            <input type="checkbox" id="explain-check" ${o?`checked`:``}>
            <span>Enable Token Attribution (Leave-one-out explainability)</span>
          </label>
        </div>

        <button id="analyze-btn" class="btn-primary" ${c?`disabled`:``}>
          ${c?`Processing Inference...`:`Analyze Review`}
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
  `;let u=document.getElementById(`domain-select`);u.addEventListener(`change`,()=>{r=u.value;let n=document.getElementById(`review-input`);n.value=t[r][0].text,p(e)});let d=document.getElementById(`model-select`);d.addEventListener(`change`,()=>{i=d.value});let f=document.getElementById(`thresh-slider`);f.addEventListener(`input`,()=>{a=parseFloat(f.value),document.getElementById(`thresh-val`).textContent=`${Math.round(a*100)}%`});let h=document.getElementById(`explain-check`);h.addEventListener(`change`,()=>{o=h.checked}),document.querySelectorAll(`.sample-chip`).forEach(e=>{e.addEventListener(`click`,e=>{let n=parseInt(e.currentTarget.dataset.idx),i=document.getElementById(`review-input`);i.value=t[r][n].text})}),document.getElementById(`analyze-btn`).addEventListener(`click`,m),s&&g(s)}async function m(){let t=document.getElementById(`review-input`).value.trim();if(!t)return;c=!0;let n=document.getElementById(`analyze-btn`);n&&(n.disabled=!0,n.textContent=`Processing Inference...`);try{let n=await fetch(`${e}/predict`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({review:t,model:i,domain:r,explain:o,confidence_threshold:a})});if(!n.ok)throw Error(`API request failed`);let c=await n.json();s=c,g(c)}catch{let e=document.getElementById(`results-viewport`);e.innerHTML=`
      <div style="background: var(--status-neg-bg); border: 1px solid var(--status-neg-border); padding: 1.25rem; border-radius: 8px; color: var(--status-neg-text);">
        <strong>API Disconnected</strong><br>
        Ensure the backend server is running in PowerShell via: <code>uvicorn api.main:app --reload</code>
      </div>
    `}finally{c=!1,n&&(n.disabled=!1,n.textContent=`Analyze Review`)}}function h(e,t){let n=[];for(let r of t){let t=r.char_start,i=r.char_end;if(t===void 0||i===void 0||t<0||i<=t||i>e.length){let n=e.toLowerCase().indexOf(r.aspect.toLowerCase());n>=0&&(t=n,i=n+r.aspect.length)}t!==void 0&&i!==void 0&&t>=0&&i<=e.length&&i>t&&(n.some(e=>Math.max(t,e.start)<Math.min(i,e.end))||n.push({start:t,end:i,sentiment:r.sentiment}))}n.sort((e,t)=>t.start-e.start);let r=e;for(let e of n){let t=r.slice(0,e.start),n=r.slice(e.start,e.end),i=r.slice(e.end);r=`${t}<span class="hl-${e.sentiment}">${n}</span>${i}`}return r}function g(e){let t=document.getElementById(`latency-indicator`);t&&(t.textContent=`${e.latency_ms} ms • ${e.model.toUpperCase()}`);let n=document.getElementById(`results-viewport`);if(!e.aspects||e.aspects.length===0){n.innerHTML=`
      <div style="text-align: center; padding: 3rem 1rem; color: var(--text-muted);">
        No aspects detected meeting the ${Math.round(a*100)}% confidence threshold.
      </div>
    `;return}let r=`
    <div class="highlight-panel">
      ${h(e.review,e.aspects)}
    </div>
    <div class="aspects-list">
  `;for(let t of e.aspects)r+=`
      <div class="aspect-card ${t.sentiment}">
        <div class="aspect-top">
          <span class="aspect-title">${t.aspect}</span>
          <span class="badge-sentiment ${t.sentiment}">
            ● ${t.sentiment} (${Math.round(t.confidence*100)}%)
          </span>
        </div>
        <div class="meter-bg">
          <div class="meter-fill ${t.sentiment}" style="width: ${Math.round(t.confidence*100)}%"></div>
        </div>
    `,t.explanation&&t.explanation.top_influential_words&&(r+=`
        <div class="explain-section">
          <span class="explain-label">Influential Tokens (Feature Attribution):</span>
          <div>
            ${t.explanation.top_influential_words.map(e=>`
              <span class="tag-word ${e.importance>=0?`support`:`oppose`}">
                ${e.word} (${e.importance>0?`+`:``}${e.importance.toFixed(2)})
              </span>
            `).join(``)}
          </div>
        </div>
      `),r+=`</div>`;r+=`
    </div>
    <div class="export-bar">
      <button class="btn-secondary" id="export-json">Export JSON</button>
      <button class="btn-secondary" id="export-csv">Export CSV</button>
    </div>
  `,n.innerHTML=r,document.getElementById(`export-json`)?.addEventListener(`click`,()=>{y(JSON.stringify(e,null,2),`absa_prediction.json`,`application/json`)}),document.getElementById(`export-csv`)?.addEventListener(`click`,()=>{let t=[`Aspect,Sentiment,Confidence,Review`];for(let n of e.aspects)t.push(`"${n.aspect}","${n.sentiment}",${n.confidence},"${e.review.replace(/"/g,`""`)}"`);y(t.join(`
`),`absa_prediction.csv`,`text/csv`)})}function _(t){t.innerHTML=`
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
  `,document.getElementById(`batch-submit-btn`).addEventListener(`click`,async()=>{let t=document.getElementById(`batch-textarea`).value.split(`
`).map(e=>e.trim()).filter(e=>e.length>0);if(t.length===0)return;let n=document.getElementById(`batch-submit-btn`);n.disabled=!0,n.textContent=`Processing Batch...`;let r=document.getElementById(`batch-domain`).value,i=document.getElementById(`batch-model`).value;try{let n=await(await fetch(`${e}/predict/batch`,{method:`POST`,headers:{"Content-Type":`application/json`},body:JSON.stringify({reviews:t,model:i,domain:r,confidence_threshold:a})})).json(),o=document.getElementById(`batch-results-viewport`),s=`
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
          <h3 style="font-size: 1.05rem; font-weight: 700;">Results (${n.total_reviews} reviews analyzed in ${n.latency_ms} ms)</h3>
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
      `;n.results.forEach((e,t)=>{s+=`
          <tr>
            <td style="color: var(--text-muted); font-weight: 700;">${t+1}</td>
            <td style="font-weight: 500;">${e.review}</td>
            <td>
              ${e.aspects.map(e=>`
                <span class="badge-sentiment ${e.sentiment}" style="margin: 2px; display: inline-flex;">
                  ${e.aspect}: ${e.sentiment}
                </span>
              `).join(``)||`<span style="color: var(--text-muted); font-size: 0.85rem;">No aspects detected</span>`}
            </td>
          </tr>
        `}),s+=`</tbody></table>`,o.innerHTML=s}catch{alert(`Batch analysis request failed.`)}finally{n.disabled=!1,n.textContent=`Process Batch`}})}function v(e){e.innerHTML=`
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
  `}function y(e,t,n){let r=new Blob([e],{type:n}),i=URL.createObjectURL(r),a=document.createElement(`a`);a.href=i,a.download=t,a.click(),URL.revokeObjectURL(i)}l();