const tabs = document.querySelectorAll('.tab');
tabs.forEach(button => button.addEventListener('click', () => {
  tabs.forEach(tab => tab.classList.toggle('active', tab === button));
  document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === button.dataset.tab));
}));

const textarea = document.querySelector('#textInput');
textarea.addEventListener('input', () => document.querySelector('#charCount').textContent = `${textarea.value.length} ký tự`);
document.querySelector('#fileInput').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.txt')) { alert('Vui lòng chọn tệp .txt'); event.target.value = ''; return; }
  textarea.value = await file.text();
  textarea.dispatchEvent(new Event('input'));
  document.querySelector('#fileLabel').textContent = file.name;
});
document.querySelectorAll('.model-option input').forEach(input => input.addEventListener('change', () => {
  document.querySelectorAll('.model-option').forEach(option => option.classList.toggle('selected', option.contains(input) && input.checked));
}));

document.querySelector('#classifyBtn').addEventListener('click', async event => {
  const button = event.currentTarget;
  const box = document.querySelector('#result');
  const model = document.querySelector('input[name="model"]:checked').value;
  button.disabled = true;
  button.innerHTML = 'Đang chạy mô hình… <span>◌</span>';
  box.innerHTML = '<div class="result-empty"><span class="result-mark">◌</span><div><strong>Đang phân tích văn bản</strong><small>Lần đầu tải mô hình có thể mất một chút thời gian.</small></div></div>';
  try {
    const response = await fetch('/api/classify', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({model,text:textarea.value})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Có lỗi xảy ra.');
    const names = {svm:'Linear SVM',bert:'BERT Base',gpt2:'GPT-2'};
    box.innerHTML = `<div class="result-content"><span class="result-mark">↳</span><div><div class="result-description">VĂN BẢN ĐƯỢC PHÂN VÀO NHÃN</div><div class="result-label">${escapeHtml(result.label)}</div></div><span class="result-badge">${names[model]}${result.confidence == null ? '' : `<br>${(result.confidence*100).toFixed(1)}% confidence`}</span></div>`;
  } catch(error) {
    box.innerHTML = `<div class="result-error">${escapeHtml(error.message)}</div>`;
  } finally { button.disabled = false; button.innerHTML = 'Phân loại văn bản <span>↗</span>'; }
});

function escapeHtml(value) { const el = document.createElement('span'); el.textContent = value; return el.innerHTML; }
const pct = value => value == null ? '—' : `${(value * 100).toFixed(2)}%`;
function card(model) {
  return `<button class="model-card" data-model="${model.id}"><div class="card-top"><span class="card-model">${model.name}</span><span class="card-open">↗</span></div><div class="card-accuracy">${pct(model.accuracy)}</div><div class="card-caption">accuracy · tập test</div><div class="card-status">${model.accuracy == null ? 'CHƯA CÓ METRICS.JSON' : `${model.samples.toLocaleString('vi-VN')} MẪU KIỂM THỬ`}</div></button>`;
}
function details(model) {
  const metrics = [['Accuracy',model.accuracy],['Precision · macro',model.precision],['Recall · macro',model.recall],['F1-score · macro',model.f1]];
  const metricHtml = metrics.map(([name,value]) => `<div class="metric"><div class="metric-label">${name}</div><div class="metric-value">${pct(value)}</div></div>`).join('');
  let matrixHtml = '';
  if (model.matrix) {
    const labels = model.matrix.labels;
    matrixHtml = `<div class="matrix-wrap"><div class="matrix-caption">CONFUSION MATRIX · ACTUAL × PREDICTED (${labels.length} NHÃN)</div><table class="matrix"><thead><tr><th>Thực tế ↓ / Dự đoán →</th>${labels.map(x=>`<th>${escapeHtml(x)}</th>`).join('')}</tr></thead><tbody>${model.matrix.values.map((row,i)=>`<tr><th>${escapeHtml(labels[i])}</th>${row.map((n,j)=>`<td class="${i===j?'diagonal':''}">${n.toLocaleString('vi-VN')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  const report = model.report ? `<details class="per-class"><summary>Xem precision / recall / F1 theo từng nhãn</summary><pre>${escapeHtml(model.report)}</pre></details>` : '';
  const unavailable = model.accuracy == null ? '<div class="no-data">Chưa tìm thấy kết quả đánh giá đã lưu cho mô hình này. Các chỉ số được để trống để tránh hiển thị số liệu không có nguồn.</div>' : '';
  const available = model.available ? '' : '<div class="no-data">Chưa tìm thấy tệp mô hình trong dự án.</div>';
  const sample = model.samples ? `${model.samples.toLocaleString('vi-VN')} mẫu test` : 'Không có thông tin tập test';
  return `<article class="detail-card"><div class="detail-head"><h3>${model.name}</h3><small>${sample}</small></div><div class="metric-grid">${metricHtml}</div>${unavailable}${matrixHtml}${report}${available}${model.accuracy != null && !model.matrix ? '<div class="detail-summary">Confusion matrix chưa được lưu cùng metrics.</div>' : ''}</article>`;
}
async function loadModels() {
  const container = document.querySelector('#modelCards');
  try {
    const response = await fetch('/api/models');
    const {models} = await response.json();
    container.innerHTML = models.map(card).join('');
    container.addEventListener('click', event => {
      const selected = event.target.closest('.model-card');
      if (!selected) return;
      selected.classList.toggle('selected');
      const active = [...container.querySelectorAll('.model-card.selected')].map(el=>models.find(m=>m.id===el.dataset.model));
      document.querySelector('#detailGrid').innerHTML = active.map(details).join('');
    });
  } catch(error) { container.innerHTML = `<p class="error">Không tải được danh sách mô hình: ${escapeHtml(error.message)}</p>`; }
}
loadModels();
