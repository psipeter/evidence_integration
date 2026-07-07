/**
 * plugin-iti-clock.js
 * jsPsych 8 plugin — circular countdown clock that auto-advances.
 * If timed_out=true, briefly shows "too slow" before the clock starts.
 */

const info = {
  name: 'iti-clock',
  parameters: {
    duration_ms: { type: 'INT',     default: 1000 },
    color:       { type: 'STRING',  default: '#2563eb' },
    radius:      { type: 'INT',     default: 60 },
    timed_out:          { type: 'BOOLEAN', default: false },
    timeouts_remaining: { type: 'INT',     default: 3 },
    iti_condition:      { type: 'STRING',  default: 'control'    }, // 'control' | 'distract'
    distractor_type:    { type: 'STRING',  default: 'none'       }, // 'none' | 'iti_length' | 'popup'
    is_binary:          { type: 'BOOLEAN', default: false        },
  },
};

class ItiClockPlugin {
  constructor(jsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_el, trial) {
    document.body.style.backgroundColor = '#f5f5f5';

    const { duration_ms, color, radius, timed_out, timeouts_remaining,
            iti_condition, distractor_type, is_binary } = trial;
    const strokeWidth = 5;
    const size = (radius + strokeWidth) * 2;
    const cx   = size / 2;
    const cy   = size / 2;

    // Guard: once the trial has finished, don't touch the DOM
    let active = true;
    let rafId  = null;
    let timeoutId = null;


    // ── Popup distractor ──────────────────────────────────────────────────────
    // Spawns random obs popups in fixed positions while the clock runs.
    // Cleaned up on finish().
    const POPUP_INTERVAL_MS = duration_ms * 0.1;  // new popup every 10% of ITI
    const POPUP_MAX         = 30;    // cap to avoid overflow
    const POPUP_FADE_MS     = 200;
    const popups            = [];
    let   popupTimer        = null;

    const _circleSize = () => {
      const vw = window.innerWidth;
      return Math.min(Math.max(60, vw * 0.11), 160);  // clamp(60px,11vw,160px)
    };

    const _numFontPx = () => {
      const vw = window.innerWidth;
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
      return Math.min(Math.max(3 * rem, vw * 0.10), 8 * rem);
    };

    const _placeNoOverlap = (w, h, existing, maxTries = 60) => {
      const margin  = 20;
      const vw      = window.innerWidth;
      const vh      = window.innerHeight;
      // Exclude the ITI clock which is centered; add generous padding
      const clkPad  = size / 2 + margin * 3;
      const clkRect = { x: vw/2 - clkPad, y: vh/2 - clkPad,
                        w: clkPad * 2,     h: clkPad * 2 };
      const maxX = vw - w - margin;
      const maxY = vh - h - margin;
      if (maxX <= margin || maxY <= margin) return null;
      for (let i = 0; i < maxTries; i++) {
        const x = margin + Math.random() * (maxX - margin);
        const y = margin + Math.random() * (maxY - margin);
        const blocked = [clkRect, ...existing].some(r =>
          x < r.x + r.w + margin && x + w > r.x - margin &&
          y < r.y + r.h + margin && y + h > r.y - margin
        );
        if (!blocked) return { x, y };
      }
      return null;
    };

    const _spawnPopup = () => {
      if (!active || popups.length >= POPUP_MAX) return;
      const el = document.createElement('div');
      // No CSS transition — remove instantly on cleanup to avoid bleed-through
      el.style.cssText = 'position:fixed;z-index:500;pointer-events:none;opacity:1;';

      let rect;
      if (is_binary) {
        const isBlue = Math.random() < 0.5;
        const sz     = _circleSize();
        el.style.cssText += `width:${sz}px;height:${sz}px;border-radius:50%;`
                          + `background:${isBlue ? '#2563eb' : '#ef4444'};`
                          + 'box-shadow:0 2px 8px rgba(0,0,0,0.2);';
        const pos = _placeNoOverlap(sz, sz, popups);
        if (!pos) return;
        el.style.left = pos.x + 'px'; el.style.top = pos.y + 'px';
        rect = { x: pos.x, y: pos.y, w: sz, h: sz };
      } else {
        const val    = Math.floor(Math.random() * 101);
        const fontPx = _numFontPx();
        const w = fontPx * 2.5, h = fontPx * 1.2;
        el.style.cssText += `font-size:${fontPx}px;font-weight:bold;color:#ef4444;`
                          + 'line-height:1;white-space:nowrap;';
        el.textContent = String(val);
        const pos = _placeNoOverlap(w, h, popups);
        if (!pos) return;
        el.style.left = pos.x + 'px'; el.style.top = pos.y + 'px';
        rect = { x: pos.x, y: pos.y, w, h };
      }

      document.body.appendChild(el);
      popups.push({ x: rect.x, y: rect.y, w: rect.w, h: rect.h, el });
    };

    const _startPopups = () => {
      _spawnPopup();
      popupTimer = setInterval(_spawnPopup, POPUP_INTERVAL_MS);
    };

    const _stopPopups = () => {
      if (popupTimer !== null) { clearInterval(popupTimer); popupTimer = null; }
      // Remove immediately — no fade, prevents bleed-through to next screen
      popups.forEach(p => { if (p.el.parentNode) p.el.parentNode.removeChild(p.el); });
      popups.length = 0;
    };

    const finish = () => {
      if (!active) return;
      active = false;
      if (rafId)     { cancelAnimationFrame(rafId); rafId = null; }
      if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; }
      _stopPopups();
      this.jsPsych.finishTrial({ screen: 'iti', duration_ms, timed_out });
    };

    const showClock = () => {
      if (!active) return;
      display_el.innerHTML = `
        <div class="iti-wrap">
          <canvas id="iti-canvas" width="${size}" height="${size}"></canvas>
        </div>`;
      if (iti_condition === 'distract' && distractor_type === 'popup') {
        _startPopups();
      }

      const canvas = display_el.querySelector('#iti-canvas');
      const ctx    = canvas.getContext('2d');
      const start  = performance.now();

      const draw = (now) => {
        if (!active) return;
        const fraction = Math.min((now - start) / duration_ms, 1);
        const endAngle = -Math.PI / 2 + fraction * 2 * Math.PI;

        ctx.clearRect(0, 0, size, size);

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth   = strokeWidth;
        ctx.stroke();

        if (fraction > 0) {
          ctx.beginPath();
          ctx.arc(cx, cy, radius, -Math.PI / 2, endAngle);
          ctx.strokeStyle = color;
          ctx.lineWidth   = strokeWidth;
          ctx.lineCap     = 'round';
          ctx.stroke();
        }

        if (fraction < 1) {
          rafId = requestAnimationFrame(draw);
        } else {
          finish();
        }
      };

      rafId = requestAnimationFrame(draw);
    };

    if (timed_out) {
      // Fade in(100ms) → out(1000ms) → in(2000ms) → hold → clock(3200ms)
      // Each transition: 0.8s. Total: 100+800+200+800+200+800+400 = 3300ms
      const FADE = 800;   // matches transition duration
      const displayMs = 3200;
      display_el.innerHTML = `
        <div class="iti-wrap" style="flex-direction:column;gap:1.2rem;">
          <span style="font-size:3rem;font-weight:bold;color:#ef4444;">
            Too slow
          </span>
          <span id="too-slow-pulse" data-timeouts-remaining="${timeouts_remaining}" style="
            font-size:2rem;font-style:italic;color:#555;
            opacity:0;transition:opacity ${FADE}ms ease;">
            ${timeouts_remaining} timeout${timeouts_remaining === 1 ? '' : 's'} remaining
          </span>
        </div>`;
      const el = display_el.querySelector('#too-slow-pulse');
      setTimeout(() => { if (el && active) el.style.opacity = '1'; }, 100);        // fade in
      setTimeout(() => { if (el && active) el.style.opacity = '0'; }, 100+FADE+200); // fade out
      setTimeout(() => { if (el && active) el.style.opacity = '1'; }, 100+FADE*2+400); // fade in
      setTimeout(() => { if (active) showClock(); }, displayMs);
    } else {
      showClock();
    }
  }
}

ItiClockPlugin.info = info;
export default ItiClockPlugin;
