/**
 * slider-continuous.js — shared slider component for the continuous task.
 *
 * Two modes (controlled by `unset`):
 *   unset=true  ('none') — thumb hidden, submit disabled until first interaction
 *   unset=false ('last') — thumb at initPos, submit immediately enabled
 *
 * initSlider() must be called after on_load() in the plugin's trial()
 * method (trial() must NOT be async — see plugin-observation-continuous.js
 * header for why) — this guarantees the DOM is fully rendered before
 * listeners attach. No setTimeout or rAF deferral needed.
 * Desktop/mouse only (Prolific restriction) — using 'click' for submit.
 */

export const SLIDER_MIN = 0;
export const SLIDER_MAX = 100;

// ── HTML ──────────────────────────────────────────────────────────────────────

export const buildSliderHTML = ({
  unset     = true,
  initPos   = 0,
  showValue = false,
} = {}) => `
  <div class="slider-section">
    <div class="slider-label-float-wrap">
      <div id="slider-float-label" class="slider-float-label"
           style="display:${!unset && showValue ? 'block' : 'none'};">
        ${!unset ? initPos : ''}
      </div>
    </div>
    <div class="slider-wrap">
      <div class="slider-track-wrap">
        <input type="range" id="response-slider"
               min="${SLIDER_MIN}" max="${SLIDER_MAX}"
               value="${initPos}" step="1"
               class="${unset ? 'slider-unset' : 'slider-last'}">
        <div class="slider-ruler" id="slider-ruler"></div>
      </div>
    </div>
  </div>`;

// ── Ruler ─────────────────────────────────────────────────────────────────────

const buildRuler = (slider) => {
  const ruler = slider.closest('.slider-track-wrap')?.querySelector('.slider-ruler');
  if (!ruler) return;
  const pctOf = v => ((v - SLIDER_MIN) / (SLIDER_MAX - SLIDER_MIN) * 100).toFixed(3) + '%';
  let html = '';
  for (let v = SLIDER_MIN; v <= SLIDER_MAX; v += 25) {
    const pct     = pctOf(v);
    const isMajor = v % 50 === 0;
    html += `<div class="slider-tick ${isMajor ? 'slider-tick-major' : ''}" style="left:${pct};height:${isMajor ? 8 : 5}px;"></div>`;
    html += `<div class="slider-tick-label" style="left:${pct};">${v}</div>`;
  }
  ruler.innerHTML = html;
};

// ── Float label ───────────────────────────────────────────────────────────────

export const updateFloatLabel = (slider) => {
  const lbl = slider.closest('.slider-section')?.querySelector('#slider-float-label');
  if (!lbl) return;
  lbl.style.display = 'block';
  lbl.textContent   = slider.value;
  const thumbR   = 3;
  const rect     = slider.getBoundingClientRect();
  const usable   = rect.width - 2 * thumbR;
  const pct      = (slider.value - slider.min) / (slider.max - slider.min);
  const px       = thumbR + pct * usable;
  const wrapLeft = slider.closest('.slider-section').getBoundingClientRect().left;
  const lblHalf  = lbl.offsetWidth / 2;
  const rawLeft  = rect.left - wrapLeft + px;
  const wrapW    = slider.closest('.slider-section').getBoundingClientRect().width;
  const clamped  = Math.max(lblHalf, Math.min(rawLeft, wrapW - lblHalf));
  lbl.style.left      = clamped + 'px';
  lbl.style.transform = 'translateX(-50%)';
};

// ── Wire-up ───────────────────────────────────────────────────────────────────
// Called after on_load() in plugin's trial() — no deferral needed.

export const initSlider = (display_el, {
  unset     = true,
  showValue = false,
  onFinish,
} = {}) => {
  const slider = display_el.querySelector('#response-slider');
  const btn    = display_el.querySelector('#submit-btn');
  if (!slider || !btn) return;

  buildRuler(slider);
  if (!unset && showValue) updateFloatLabel(slider);

  slider.addEventListener('mousedown', () => {
    slider.classList.remove('slider-unset');
    slider.classList.remove('slider-last');
    btn.disabled = false;
    requestAnimationFrame(() => { if (showValue) updateFloatLabel(slider); });
  });

  slider.addEventListener('input', () => {
    slider.classList.remove('slider-unset');
    slider.classList.remove('slider-last');
    btn.disabled = false;
    if (showValue) updateFloatLabel(slider);
  });

  // 'click' is reliable for desktop/mouse — fires after mousedown+mouseup on same element
  btn.addEventListener('click', (e) => {
    if (!btn.disabled) {
      e.preventDefault();
      onFinish();
    }
  });

  if (showValue && typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => {
      if (!slider.classList.contains('slider-unset')) updateFloatLabel(slider);
    });
    ro.observe(slider);
  }
};
