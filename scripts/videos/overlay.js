// On-screen layer for how-to recordings: a pointer, a highlight ring, a caption
// bar, an address pill, and full-screen title cards. record.py calls these
// through window.howto after each page load.
(() => {
  const css = `
  #howto-layer { position: fixed; inset: 0; pointer-events: none; z-index: 2147483647;
    font-family: "Open Sans", "Segoe UI", Arial, sans-serif; }
  #howto-pointer { position: absolute; left: 640px; top: 400px; width: 28px; height: 28px;
    transition: left .7s cubic-bezier(.4,0,.2,1), top .7s cubic-bezier(.4,0,.2,1), transform .15s;
    filter: drop-shadow(0 2px 3px rgba(0,0,0,.35)); }
  #howto-pointer.press { transform: scale(.82); }
  #howto-ring { position: absolute; border: 3px solid #f2a900; border-radius: 10px;
    box-shadow: 0 0 0 9999px rgba(10,22,40,.28); opacity: 0; transition: all .35s ease; }
  #howto-ring.on { opacity: 1; }
  #howto-caption { position: absolute; left: 50%; bottom: 28px; transform: translateX(-50%);
    max-width: 900px; padding: 12px 22px; border-radius: 10px; background: rgba(0,40,85,.92);
    color: #fff; font-size: 22px; line-height: 1.35; text-align: center; opacity: 0;
    transition: opacity .3s; }
  #howto-caption.on { opacity: 1; }
  #howto-url { position: absolute; left: 50%; top: 70px; transform: translateX(-50%);
    padding: 8px 16px; border-radius: 999px; background: #fff; color: #002855;
    border: 2px solid #f2a900; font: 600 17px ui-monospace, Menlo, Consolas, monospace;
    box-shadow: 0 4px 14px rgba(0,0,0,.18); white-space: nowrap; opacity: 0; transition: opacity .3s; }
  #howto-url.on { opacity: 1; }
  #howto-spot { position: absolute; inset: 0; background: rgba(10,22,40,.55); display: none;
    align-items: center; justify-content: center; }
  #howto-spot.on { display: flex; }
  #howto-spot img { width: 1180px; max-height: 600px; object-fit: contain; background: #fff;
    border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,.35); margin-bottom: 40px; }
  #howto-card { position: absolute; inset: 0; background: #002855; color: #fff;
    display: flex; flex-direction: column; justify-content: center; padding: 0 110px;
    opacity: 0; transition: opacity .4s; }
  #howto-card.on { opacity: 1; }
  #howto-card .kicker { color: #f2a900; font-size: 20px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; }
  #howto-card .title { font-size: 54px; font-weight: 700; line-height: 1.15; margin-top: 14px; }
  #howto-card .sub { font-size: 24px; margin-top: 22px; color: #c9d6e8; max-width: 900px; }
  `;
  const pointer = `<svg viewBox="0 0 24 24" width="28" height="28"><path d="M3 2l7 19 2.6-7.4L20 11z"
    fill="#fff" stroke="#002855" stroke-width="1.6" stroke-linejoin="round"/></svg>`;

  let state = {x: 640, y: 400};
  function layer() {
    let el = document.getElementById('howto-layer');
    if (el) return el;
    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
    el = document.createElement('div');
    el.id = 'howto-layer';
    el.innerHTML = `<div id="howto-spot"></div><div id="howto-ring"></div><div id="howto-url"></div>
      <div id="howto-caption"></div><div id="howto-card"></div>
      <div id="howto-pointer">${pointer}</div>`;
    document.body.appendChild(el);
    const p = el.querySelector('#howto-pointer');
    p.style.transition = 'none';
    p.style.left = state.x + 'px';
    p.style.top = state.y + 'px';
    requestAnimationFrame(() => { p.style.transition = ''; });
    return el;
  }
  const $ = (id) => layer().querySelector('#' + id);

  window.howto = {
    point(box) {
      if (!box) return;
      state = {x: box.x + Math.min(box.width / 2, 60), y: box.y + box.height / 2};
      $('howto-pointer').style.left = state.x + 'px';
      $('howto-pointer').style.top = state.y + 'px';
    },
    ring(box) {
      const r = $('howto-ring');
      if (!box) { r.classList.remove('on'); return; }
      const pad = 6;
      Object.assign(r.style, {left: box.x - pad + 'px', top: box.y - pad + 'px',
        width: box.width + 2 * pad + 'px', height: box.height + 2 * pad + 'px'});
      r.classList.add('on');
    },
    press() {
      const p = $('howto-pointer');
      p.classList.add('press');
      setTimeout(() => p.classList.remove('press'), 180);
    },
    caption(text) {
      const c = $('howto-caption');
      c.textContent = text || '';
      c.classList.toggle('on', Boolean(text));
    },
    url(on) {
      const u = $('howto-url');
      u.textContent = location.host + location.pathname + location.hash;
      u.classList.toggle('on', on);
    },
    // Shows an image enlarged over the page. A new image restarts its animation;
    // the same source again leaves the running image in place.
    spot(src) {
      const sp = $('howto-spot');
      if (!src) { sp.classList.remove('on'); sp.innerHTML = ''; sp.dataset.src = ''; return; }
      if (sp.dataset.src !== src) {
        sp.dataset.src = src;
        sp.innerHTML = `<img src="${src.split('?')[0]}?t=${Date.now()}">`;
      }
      sp.classList.add('on');
    },
    card(spec) {
      const c = $('howto-card');
      if (!spec) { c.classList.remove('on'); $('howto-pointer').style.opacity = 1; return; }
      c.innerHTML = `<div class="kicker">${spec.kicker || ''}</div>
        <div class="title">${spec.title || ''}</div><div class="sub">${spec.subtitle || ''}</div>`;
      c.classList.add('on');
      $('howto-caption').classList.remove('on');
      $('howto-pointer').style.opacity = 0;
    },
  };
  if (document.body) layer();
  else document.addEventListener('DOMContentLoaded', layer);
})();
