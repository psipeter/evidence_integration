/**
 * slider-binary.js — shared binary slider component.
 *
 * Two modes:
 *   unset=true  ('none') — thumb hidden, gray track, submit disabled
 *   unset=false ('last') — thumb at initPos, submit disabled until interaction
 *
 * initBinarySlider() must be called after on_load() in the plugin's trial()
 * method (trial() must NOT be async — see plugin-observation-binary.js header
 * for why) — no setTimeout or rAF deferral needed.
 * Desktop/mouse only (Prolific restriction) — using 'click' for submit.
 */

export const BINARY_MIN = 0;
export const BINARY_MAX = 100;

const BLUE_RGB = [37, 99, 235];
const GRAY_RGB = [136, 136, 136];
const RED_RGB  = [239, 68, 68];

const lerpRgb = (t) => {
  const [a, b] = t < 0.5 ? [RED_RGB, GRAY_RGB] : [GRAY_RGB, BLUE_RGB];
  const s = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return `rgb(${Math.round(a[0]+s*(b[0]-a[0]))},${Math.round(a[1]+s*(b[1]-a[1]))},${Math.round(a[2]+s*(b[2]-a[2]))})`;
};

// ── HTML ──────────────────────────────────────────────────────────────────────

export const buildBinarySliderHTML = ({
  unset     = true,
  initPos   = 50,
  showValue = false,
} = {}) => `
  <div class="binary-slider-section">
    <div class="slider-label-float-wrap">
      <div id="slider-float-label-blue" class="slider-float-label-blue"
           style="display:${!unset && showValue ? 'block' : 'none'};">
        ${!unset ? Math.round(initPos) + '%' : ''}
      </div>
      <div id="slider-float-label-red" class="slider-float-label-red"
           style="display:${!unset && showValue ? 'block' : 'none'};">
        ${!unset ? Math.round(100 - initPos) + '%' : ''}
      </div>
    </div>
    <div class="binary-slider-row">
      <div class="slider-track-wrap" style="flex:1;">
        <input type="range" id="response-slider"
               class="binary-slider ${unset ? 'slider-unset' : 'slider-last'}"
               min="${BINARY_MIN}" max="${BINARY_MAX}"
               value="${initPos}" step="1"
               style="--pct:${unset ? '50' : initPos}%">
        <div class="slider-ruler binary-ruler" id="slider-ruler"></div>
      </div>
    </div>
  </div>`;

// ── Ruler ─────────────────────────────────────────────────────────────────────

// Each tick shows what blue%/red% WOULD BE if the slider were set to that
// position — like axis gridline numbers, not a filled-in answer, so it
// doesn't create a prior the way pre-filling the track itself did. The
// blue number's font size scales with its own value (bigger number = bigger
// text) and same for red, so the "weight" of each tick's text visually
// echoes the proportion it represents — e.g. the v=100 tick reads a large
// "100" in blue next to a tiny "0" in red.
//
// blue% == the slider's raw value (see updateBinaryLabel) and red% ==
// 100 - value, exactly mirroring the gradient/floating-label math elsewhere
// in this file, so this can't drift out of sync with them the way the
// removed directional-text ticks did.
const FONT_MIN = 0.5;    // rem, at value 0
const FONT_MAX = 2;      // rem, at value 100
const fontSizeFor = (value) => FONT_MIN + (value / 100) * (FONT_MAX - FONT_MIN);

const buildBinaryRuler = (slider) => {
  const ruler = slider.closest('.slider-track-wrap')?.querySelector('.slider-ruler');
  if (!ruler) return;
  let html = '';
  for (let v = BINARY_MIN; v <= BINARY_MAX; v += 25) {
    const pct     = v + '%';
    const isMajor = v % 50 === 0;
    html += `<div class="slider-tick ${isMajor ? 'slider-tick-major' : ''}" style="left:${pct};height:${isMajor ? 8 : 5}px;"></div>`;

    const blueVal = v;
    const redVal  = 100 - v;
    const posStyle = v === BINARY_MIN
      ? `left:0;transform:none;`
      : v === BINARY_MAX
        ? `left:auto;right:0;transform:none;`
        : `left:${pct};transform:translateX(-50%);`;

    // At the extremes, the "0" side is redundant (barely visible anyway at
    // FONT_MIN) — just show the single large dominant number.
    const pairHTML = v === BINARY_MIN
      ? `<span style="color:#ef4444;font-weight:bold;font-size:${fontSizeFor(redVal).toFixed(2)}rem;">${redVal}</span>`
      : v === BINARY_MAX
        ? `<span style="color:#2563eb;font-weight:bold;font-size:${fontSizeFor(blueVal).toFixed(2)}rem;">${blueVal}</span>`
        : `<span style="color:#2563eb;font-weight:bold;font-size:${fontSizeFor(blueVal).toFixed(2)}rem;">${blueVal}</span>` +
          `<span style="color:#bbb;font-size:0.9rem;">·</span>` +
          `<span style="color:#ef4444;font-weight:bold;font-size:${fontSizeFor(redVal).toFixed(2)}rem;">${redVal}</span>`;

    html += `<div class="slider-tick-label" style="${posStyle}display:inline-flex;align-items:baseline;gap:0.25em;white-space:nowrap;">${pairHTML}</div>`;
  }
  ruler.innerHTML = html;
};

// ── Track fill ────────────────────────────────────────────────────────────────

export const updateBinaryFill = (slider) => {
  const pct = (slider.value - slider.min) / (slider.max - slider.min) * 100;
  slider.style.setProperty('--pct', pct + '%');
};

// ── Float label (fixed positions — blue left, red right) ──────────────────────

export const updateBinaryLabel = (slider) => {
  const section = slider.closest('.binary-slider-section');
  const lblBlue = section?.querySelector('#slider-float-label-blue');
  const lblRed  = section?.querySelector('#slider-float-label-red');
  if (!lblBlue || !lblRed) return;
  lblBlue.textContent     = Math.round(slider.value) + '%';
  lblRed.textContent      = Math.round(100 - slider.value) + '%';
  lblBlue.style.display   = 'block';
  lblRed.style.display    = 'block';
  lblBlue.style.left      = '0px';
  lblBlue.style.transform = 'none';
  lblRed.style.right      = '0px';
  lblRed.style.left       = 'auto';
  lblRed.style.transform  = 'none';
};

// ── Wire-up ───────────────────────────────────────────────────────────────────
// Called after on_load() in plugin's trial() — no deferral needed.

export const initBinarySlider = (display_el, {
  unset     = true,
  showValue = false,
  onFinish,
} = {}) => {
  const slider = display_el.querySelector('#response-slider');
  const btn    = display_el.querySelector('#submit-btn');
  if (!slider || !btn) return;

  btn.disabled = true;
  buildBinaryRuler(slider);
  if (!unset && showValue) { updateBinaryLabel(slider); updateBinaryFill(slider); }

  slider.addEventListener('mousedown', () => {
    slider.classList.remove('slider-unset');
    slider.classList.remove('slider-last');
    btn.disabled = false;
    requestAnimationFrame(() => {
      updateBinaryFill(slider);
      if (showValue) updateBinaryLabel(slider);
    });
  });

  slider.addEventListener('input', () => {
    slider.classList.remove('slider-unset');
    slider.classList.remove('slider-last');
    btn.disabled = false;
    updateBinaryFill(slider);
    if (showValue) updateBinaryLabel(slider);
  });

  btn.addEventListener('click', (e) => {
    if (!btn.disabled) {
      e.preventDefault();
      onFinish();
    }
  });

  if (showValue && typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => {
      if (!slider.classList.contains('slider-unset')) updateBinaryLabel(slider);
    });
    ro.observe(slider);
  }
};
