// Helpers for the videos on the document collection of examples/rag-public.
// Load after stage.js and after local/rag-public/pages/pages.js (window.PAGES).

const PAGE_DIR = '../../../local/rag-public/pages/';
const SRC = 'Source: World Bank, Kenya Economic Update and Uganda Economic Update, 2022–2026, Open Knowledge Repository. CC BY-NC 3.0 IGO / CC BY 3.0 IGO.';
const ANSWERS_LINE = 'Answers recorded with anthropic/claude-haiku-5.5 at temperature 0.';

// The steps of RAG, in order, for stepMap() in stage.js.
const RAG_STEPS = [['Prepared once', ['Documents', 'Text', 'Passages', 'Vectors', 'Store']],
                   ['For each question', ['Question', 'Search', 'Prompt', 'Answer']]];
const ragMap = (el, at = -1) => stepMap(el, RAG_STEPS, at);

// Fill every .srcline of the page with the source line (and the answers line where data-answers is set).
function sourceLines() {
  document.querySelectorAll('.srcline').forEach((e) => {
    e.innerHTML = SRC + ('answers' in e.dataset ? '<br>' + ANSWERS_LINE : '');
  });
}

const coverOf = (id) => PAGES.covers.find((c) => c.id === id);
const citedOf = (id, page) => PAGES.cited.find((c) => c.id === id && c.page === page);

// A page image in a frame. Boxes are fractions of the page [x0, y0, x1, y1].
// Returns { inner, fit(), zoom(box, width) }; children added to `inner` are
// placed in image pixels and scale with the page.
function pageView(el, cited) {
  const W = cited.width, H = cited.height, fw = el.clientWidth, fh = el.clientHeight;
  el.insertAdjacentHTML('afterbegin', `<div class="inner"><img src="${PAGE_DIR}${cited.file}" width="${W}" height="${H}"></div>`);
  const inner = el.querySelector('.inner');
  const set = (s, tx, ty) => { inner.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`; inner.style.setProperty('--inv', 1 / s); };
  const fit = () => { const s = Math.min(fw / W, fh / H); set(s, (fw - W * s) / 2, (fh - H * s) / 2); };
  const zoom = (box, width = .82) => {
    const bw = (box[2] - box[0]) * W, s = Math.min(fw * width / bw, 3);
    const cx = (box[0] + box[2]) / 2 * W, cy = (box[1] + box[3]) / 2 * H;
    set(s, fw / 2 - cx * s, fh / 2 - cy * s);
  };
  fit();
  return { inner, fit, zoom, W, H };
}
// A box on the page, in image pixels.
function pageBox(view, box, cls, label) {
  const d = document.createElement('div');
  d.className = cls;
  Object.assign(d.style, { left: box[0] * view.W + 'px', top: box[1] * view.H + 'px',
    width: (box[2] - box[0]) * view.W + 'px', height: (box[3] - box[1]) * view.H + 'px' });
  if (label) d.innerHTML = `<b>${label}</b>`;
  view.inner.appendChild(d);
  return d;
}

// Bars for the first numbers of a vector; `of`, when given, is shown as the full length.
function vecBars(el, values, of) {
  el.innerHTML = values.map((v) => {
    const h = Math.max(3, Math.min(40, Math.abs(v) * 340));
    return `<i style="height:${h}px;margin-${v >= 0 ? 'bottom' : 'top'}:${h}px"></i>`;
  }).join('') + (of ? `<span>… ${of} numbers</span>` : '<span>…</span>');
  [...el.querySelectorAll('i')].forEach((b, k) => later(18 * k, () => b.classList.add('in2')));
}

function countUp(el, to, ms, fmt = (n) => n.toLocaleString('en-US')) {
  const t0 = performance.now(), e = epoch;
  (function tick(now) {
    if (e !== epoch) return;
    const f = Math.min(1, (now - t0) / ms);
    el.textContent = fmt(Math.round(to * (1 - Math.pow(1 - f, 3))));
    if (f < 1) requestAnimationFrame(tick);
  })(t0);
}

// Seeded random numbers, so that an illustration is the same on every run.
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}
