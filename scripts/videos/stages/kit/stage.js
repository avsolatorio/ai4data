// Stage kit for explainer videos. record.py calls go(scene, beat): an element
// with data-b="k" in section #s<scene> appears once the beat reaches k, and
// HOOKS[scene][k], defined by the stage, runs the animation of that beat.
// Timers started with later() stop when the scene changes.
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
let current = null, epoch = 0;
const later = (ms, fn) => { const e = epoch; setTimeout(() => { if (e === epoch) fn(); }, ms); };

// A cubic curve from the right edge of a to the left edge of b, drawn in
// over `ms`. Returns the path, so that dots can travel along it.
// The box of an element where it comes to rest. Elements revealed with data-b
// slide in over half a second; a wire drawn during the slide would point at
// the wrong place, so the current offset of each sliding ancestor is removed.
function rectOf(el) {
  const r = el.getBoundingClientRect();
  let dx = 0, dy = 0;
  for (let e = el; e && e !== document.body; e = e.parentElement) {
    if (!e.hasAttribute('data-b')) continue;
    const m = new DOMMatrixReadOnly(getComputedStyle(e).transform);
    dx += m.m41; dy += m.m42;
  }
  return { left: r.left - dx, right: r.right - dx, top: r.top - dy, bottom: r.bottom - dy, width: r.width, height: r.height };
}

function wire(svg, a, b, opts = {}) {
  const ra = rectOf(a), rb = rectOf(b), so = svg.getBoundingClientRect();
  const x1 = (opts.from === 'bottom' ? ra.left + ra.width / 2 : ra.right) - so.left;
  const y1 = (opts.from === 'bottom' ? ra.bottom : ra.top + ra.height / 2) - so.top;
  const x2 = (opts.to === 'top' ? rb.left + rb.width / 2 : rb.left) - so.left;
  const y2 = (opts.to === 'top' ? rb.top : rb.top + rb.height / 2) - so.top;
  const p = document.createElementNS(NS, 'path');
  const d = opts.from === 'bottom'
    ? `M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}`
    : `M${x1},${y1} C${x1 + (x2 - x1) * .5},${y1} ${x2 - (x2 - x1) * .5},${y2} ${x2},${y2}`;
  p.setAttribute('d', d);
  p.setAttribute('class', 'wire' + (opts.hot ? ' hot' : ''));
  if (opts.color) p.style.stroke = opts.color;
  svg.appendChild(p);
  const L = p.getTotalLength();
  p.style.strokeDasharray = L;
  p.style.strokeDashoffset = L;
  p.getBoundingClientRect();
  p.style.transition = `stroke-dashoffset ${opts.ms || 700}ms ease`;
  p.style.strokeDashoffset = 0;
  return p;
}

// Dots that travel along a path in a loop while the scene stays on.
function flow(svg, path, color, period = 1600, n = 3, r = 5) {
  const e = epoch, L = path.getTotalLength(), dots = [];
  for (let i = 0; i < n; i++) {
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('r', r); c.setAttribute('fill', color);
    svg.appendChild(c); dots.push(c);
  }
  const t0 = performance.now();
  (function tick(now) {
    if (e !== epoch) return;
    dots.forEach((c, i) => {
      const f = (((now - t0) / period) + i / n) % 1;
      const pt = path.getPointAtLength(f * L);
      c.setAttribute('cx', pt.x); c.setAttribute('cy', pt.y);
      c.setAttribute('opacity', Math.sin(Math.PI * f).toFixed(2));
    });
    requestAnimationFrame(tick);
  })(t0);
}

// A labelled token that travels once along a path; onEnd runs at arrival.
function carry(path, html, color, ms, onStep, onEnd) {
  const e = epoch, L = path.getTotalLength(), so = path.ownerSVGElement.getBoundingClientRect();
  const t = document.createElement('div');
  t.className = 'token'; t.innerHTML = html; t.style.background = color;
  current.appendChild(t);
  const t0 = performance.now();
  (function tick(now) {
    if (e !== epoch) { t.remove(); return; }
    const f = Math.min(1, (now - t0) / ms), k = f < .5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2;
    const pt = path.getPointAtLength(k * L);
    t.style.left = so.left + pt.x + 'px'; t.style.top = so.top + pt.y + 'px';
    if (onStep) onStep(f, t);
    if (f < 1) requestAnimationFrame(tick); else { t.style.transition = 'opacity .3s'; t.style.opacity = 0; setTimeout(() => t.remove(), 300); if (onEnd) onEnd(); }
  })(t0);
}

// A chip that moves from one element to another (screen coordinates).
function fly(fromEl, toEl, html, ms, cls = '') {
  const a = rectOf(fromEl), b = rectOf(toEl);
  const t = document.createElement('div');
  t.className = 'token ' + cls; t.innerHTML = html; t.style.background = 'var(--primary)';
  current.appendChild(t);
  const kf = [{ left: a.left + a.width / 2 + 'px', top: a.top + a.height / 2 + 'px' },
              { left: b.left + b.width / 2 + 'px', top: b.top + b.height / 2 + 'px' }];
  t.animate(kf, { duration: ms, easing: 'cubic-bezier(.4,0,.2,1)', fill: 'forwards' }).onfinish = () => {
    t.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 250, fill: 'forwards' }).onfinish = () => t.remove();
  };
}

function typeText(el, text, cps, done) {
  let i = 0; el.classList.add('caret');
  const step = () => { el.textContent = text.slice(0, ++i); if (i < text.length) later(1000 / cps, step); else { el.classList.remove('caret'); if (done) done(); } };
  step();
}

// Streams tokens into el. A token is a string of HTML; tokens are appended
// in order, every `ms`. onToken(i) runs after each one.
function stream(el, tokens, ms, onToken, done) {
  let i = 0; el.classList.add('caret');
  const step = () => {
    el.insertAdjacentHTML('beforeend', tokens[i]);
    if (onToken) onToken(i);
    i++;
    if (i < tokens.length) later(ms * (tokens[i - 1].length > 30 ? 1.6 : 1), step); else { el.classList.remove('caret'); if (done) done(); }
  };
  step();
}
const words = (s) => s.match(/\S+\s*/g);

const reached = {};
window.go = (scene, beat = 0) => {
  const key = String(scene), s = $('s' + key);
  if (current !== s) {
    epoch++;
    if (current) current.classList.remove('on');
    s.classList.add('on');
    current = s;
    reached[key] = 0;
  }
  s.querySelectorAll('[data-b]').forEach((el) => el.classList.toggle('in', Number(el.dataset.b) <= beat));
  // Hooks run once, in order, for each beat reached since the last call.
  for (let b = (reached[key] || 0) + 1; b <= beat; b++) {
    const h = HOOKS[key] && HOOKS[key][b];
    if (h) later(b === beat ? 120 : 0, h);
  }
  reached[key] = Math.max(reached[key] || 0, beat);
};

// A map of the steps of a method: groups of [name, [step, ...]]. mapAt(el, i)
// marks the steps before i as done and step i as current (steps are counted
// across the groups).
const ARROW = '<svg class="ar" viewBox="0 0 14 14"><path d="M4 2l5 5-5 5" fill="none" stroke="currentColor" stroke-width="2"/></svg>';
function stepMap(el, groups, at = -1) {
  let i = 0;
  el.innerHTML = groups.map(([name, steps], g) =>
    (g ? '<i class="sep"></i>' : '') + `<div class="grp">${name ? `<span>${name}</span>` : ''}` +
    steps.map((s, k) => (k ? ARROW : '') + `<b class="st" data-i="${i++}">${s}</b>`).join('') + '</div>').join('');
  mapAt(el, at);
}
function mapAt(el, i) {
  el.querySelectorAll('.st').forEach((s) => {
    const k = Number(s.dataset.i);
    s.classList.toggle('done', k < i);
    s.classList.toggle('on', k === i);
  });
}

// A copy of `html` that moves from the centre of one element to the centre of
// another at its own size, then fades. Stretching a small element to the size
// of a large one distorts it, so the copy keeps its size and only moves.
function moveCard(fromEl, toEl, html, ms, onEnd, cls = '') {
  const a = rectOf(fromEl), b = rectOf(toEl), sb = current.getBoundingClientRect();
  const m = document.createElement('div');
  m.className = 'mover-card ' + cls; m.innerHTML = html;
  current.appendChild(m);
  const w = m.offsetWidth, h = m.offsetHeight;
  const x0 = a.left + a.width / 2 - w / 2 - sb.left, y0 = a.top + a.height / 2 - h / 2 - sb.top;
  const x1 = b.left + b.width / 2 - w / 2 - sb.left, y1 = b.top + b.height / 2 - h / 2 - sb.top;
  m.style.left = x0 + 'px'; m.style.top = y0 + 'px';
  m.animate([{ transform: 'translate(0,0)', opacity: 0 }, { opacity: 1, offset: .15 },
             { transform: `translate(${x1 - x0}px, ${y1 - y0}px)`, opacity: 1 }],
            { duration: ms, easing: 'cubic-bezier(.4,0,.2,1)', fill: 'forwards' }).onfinish = () => {
    if (onEnd) onEnd();
    m.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 250, fill: 'forwards' }).onfinish = () => m.remove();
  };
}

