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

// Two separate rows instead of one row of blue·red pairs — pairing them on
// the same row/position was reported as confusing. Row 1 (closer to the
// track) shows all 5 blue values 0,25,...,100; row 2 (below it) shows the
// mirrored red values 100,75,...,0, at the SAME x positions as row 1 so the
// two rows read as a matched pair vertically rather than side-by-side.
// All numbers share one fixed size — an earlier version scaled font size by
// each number's own value, but that made the rows harder to scan as a
// simple axis; a uniform size reads more like a normal ruler.
const TICK_FONT_SIZE = 1.3; // rem, all tick numbers

const ROW1_TOP = 8;    // px — blue row
const ROW2_TOP = 40;   // px — red row, below blue row

const buildBinaryRuler = (slider) => {
  const ruler = slider.closest('.slider-track-wrap')?.querySelector('.slider-ruler');
  if (!ruler) return;
  let html = '';

  for (let v = BINARY_MIN; v <= BINARY_MAX; v += 25) {
    const pct     = v + '%';
    const isMajor = v % 50 === 0;
    html += `<div class="slider-tick ${isMajor ? 'slider-tick-major' : ''}" style="left:${pct};height:${isMajor ? 8 : 5}px;"></div>`;
  }

  const buildRow = (getValue, color, top) => {
    let row = '';
    for (let v = BINARY_MIN; v <= BINARY_MAX; v += 25) {
      const pct = v + '%';
      const val = getValue(v);
      const posStyle = v === BINARY_MIN
        ? `left:0;transform:none;`
        : v === BINARY_MAX
          ? `left:auto;right:0;transform:none;`
          : `left:${pct};transform:translateX(-50%);`;
      row += `<div style="position:absolute;top:${top}px;${posStyle}` +
             `color:${color};font-weight:bold;line-height:1;white-space:nowrap;` +
             `font-size:${TICK_FONT_SIZE}rem;">${val}</div>`;
    }
    return row;
  };

  html += buildRow(v => v,       '#2563eb', ROW1_TOP);
  html += buildRow(v => 100 - v, '#ef4444', ROW2_TOP);

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

// ── v2 layout (experimental) ──────────────────────────────────────────────────
// Blue band ABOVE the slider track, red band BELOW it -- PURELY STATIC axis
// (just the tick marks + reference numbers 0/25/50/75/100, no moving
// content at all). Kept fully separate from the original
// buildBinarySliderHTML/initBinarySlider above (unchanged) -- this is a
// deliberate A/B: switching a plugin's import back to the original
// functions reverts completely, no digging through git history needed.
//
// Design history/why the dynamic value moved OUT of this axis entirely:
// two earlier attempts put a moving "current value" badge in this same
// axis space -- first centered over the ticks (which pushed the tick marks
// away from the slider AND left the badge too close to the thumb at the
// same time), then offset toward the numbers side with a small occluding
// background (which still let a sliver of the numbers/ticks leak out from
// under it, since the occluding box's height was only an approximation of
// what needed covering). Rather than keep tuning that occlusion, the
// dynamic value now lives INSIDE the colored bar itself, flanking the
// thumb directly (see buildBinaryInBarValues below) -- this axis goes back
// to being purely the static calibration guide (constraint #3), with no
// moving parts to keep in sync with anything else.
const buildBandInnerHTML = (getValue, color) => {
  let numbersHtml = '', ticksHtml = '';
  for (let v = BINARY_MIN; v <= BINARY_MAX; v += 25) {
    const pct     = v + '%';
    const isMajor = v % 50 === 0;
    const val     = getValue(v);
    const posStyle = v === BINARY_MIN
      ? `left:0;transform:none;`
      : v === BINARY_MAX
        ? `left:auto;right:0;transform:none;`
        : `left:${pct};transform:translateX(-50%);`;
    numbersHtml += `<div class="binary-ruler-num" style="${posStyle}color:${color};">${val}</div>`;
    ticksHtml   += `<div class="binary-ruler-tick ${isMajor ? 'binary-ruler-tick-major' : ''}" style="left:${pct};"></div>`;
  }
  return `
    <div class="binary-ruler-inner">
      <div class="binary-ruler-numbers">${numbersHtml}</div>
      <div class="binary-ruler-ticks">${ticksHtml}</div>
    </div>`;
};

const buildBinaryRulerV2 = (display_el) => {
  const top    = display_el.querySelector('#slider-ruler-top');
  const bottom = display_el.querySelector('#slider-ruler-bottom');
  if (!top || !bottom) return;
  top.innerHTML    = buildBandInnerHTML(v => v,       '#2563eb');
  bottom.innerHTML = buildBandInnerHTML(v => 100 - v, '#ef4444');
};

// ── In-bar dynamic values ──────────────────────────────────────────────────────
// The moving %s live INSIDE the colored bar itself, flanking the thumb --
// white bold text directly on the gradient fill, each number sitting in its
// own color's zone (blue % to the left of the thumb, red % to the right).
// Near the extremes, one zone can get too narrow to hold its own number
// without overflowing past the track edge or crowding the thumb -- rather
// than shrink/wrap/truncate the text, that number is simply not drawn at
// all once it wouldn't fit (e.g. "95%" blue shows fine with no red number
// visible at all, rather than a cramped or overflowing "5%"). Measures the
// actual rendered width of each number (not an estimate from character
// count) before deciding whether it fits, so this stays correct regardless
// of font/size changes later.
const INBAR_THUMB_HALF_WIDTH = 4;  // px, half of the thumb's own 6px-wide flat rectangle (approximate clearance around the thumb, not pixel-exact -- the thumb's true rendered position can vary slightly by browser)
const INBAR_GAP             = 8;   // px, minimum clearance between a number and the thumb

export const updateBinaryInBarValues = (slider) => {
  const wrap = slider.closest('.slider-track-wrap-v2');
  if (!wrap) return;
  const blueEl = wrap.querySelector('#binary-inbar-blue');
  const redEl  = wrap.querySelector('#binary-inbar-red');
  if (!blueEl || !redEl) return;

  const val         = Number(slider.value);
  const pct         = (val - slider.min) / (slider.max - slider.min);
  const trackWidth  = slider.getBoundingClientRect().width;
  const thumbX      = pct * trackWidth;

  blueEl.textContent = Math.round(val) + '%';
  redEl.textContent  = Math.round(100 - val) + '%';
  // Must be visible (display:block) to measure offsetWidth below -- if it
  // ends up not fitting, this gets set back to 'none' immediately after.
  blueEl.style.display = 'block';
  redEl.style.display  = 'block';

  const blueAvail = thumbX - INBAR_THUMB_HALF_WIDTH - INBAR_GAP;
  const redAvail  = trackWidth - thumbX - INBAR_THUMB_HALF_WIDTH - INBAR_GAP;

  if (blueEl.offsetWidth > blueAvail) {
    blueEl.style.display = 'none';
  } else {
    blueEl.style.left  = (thumbX - INBAR_THUMB_HALF_WIDTH - INBAR_GAP - blueEl.offsetWidth) + 'px';
    blueEl.style.right = 'auto';
  }
  if (redEl.offsetWidth > redAvail) {
    redEl.style.display = 'none';
  } else {
    redEl.style.left  = (thumbX + INBAR_THUMB_HALF_WIDTH + INBAR_GAP) + 'px';
    redEl.style.right = 'auto';
  }
};

export const buildBinarySliderHTMLv2 = ({
  unset     = true,
  initPos   = 50,
  showValue = false,
} = {}) => `
  <div class="binary-slider-section-v2">
    <div class="slider-track-wrap-v2">
      <div class="binary-ruler-band binary-ruler-band-top" id="slider-ruler-top"></div>
      <div class="binary-input-wrap">
        <input type="range" id="response-slider"
               class="binary-slider binary-slider-v2 ${unset ? 'slider-unset' : 'slider-last'}"
               min="${BINARY_MIN}" max="${BINARY_MAX}"
               value="${initPos}" step="1"
               style="--pct:${unset ? '50' : initPos}%">
        <div class="binary-inbar-values">
          <div id="binary-inbar-blue" class="binary-inbar-value" style="display:none;"></div>
          <div id="binary-inbar-red"  class="binary-inbar-value" style="display:none;"></div>
        </div>
      </div>
      <div class="binary-ruler-band binary-ruler-band-bottom" id="slider-ruler-bottom"></div>
    </div>
  </div>`;

export const initBinarySliderV2 = (display_el, {
  unset     = true,
  showValue = false,
  onFinish,
} = {}) => {
  const slider = display_el.querySelector('#response-slider');
  const btn    = display_el.querySelector('#submit-btn');
  if (!slider || !btn) return;

  btn.disabled = true;
  buildBinaryRulerV2(display_el);
  if (!unset && showValue) { updateBinaryInBarValues(slider); updateBinaryFill(slider); }

  slider.addEventListener('mousedown', () => {
    slider.classList.remove('slider-unset');
    slider.classList.remove('slider-last');
    btn.disabled = false;
    requestAnimationFrame(() => {
      updateBinaryFill(slider);
      if (showValue) updateBinaryInBarValues(slider);
    });
  });

  slider.addEventListener('input', () => {
    slider.classList.remove('slider-unset');
    slider.classList.remove('slider-last');
    btn.disabled = false;
    updateBinaryFill(slider);
    if (showValue) updateBinaryInBarValues(slider);
  });

  btn.addEventListener('click', (e) => {
    if (!btn.disabled) {
      e.preventDefault();
      onFinish();
    }
  });

  if (showValue && typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => {
      if (!slider.classList.contains('slider-unset')) updateBinaryInBarValues(slider);
    });
    ro.observe(slider);
  }
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
