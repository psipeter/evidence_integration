// Drives the sequential image-replace build on "Theories and Models" and
// "Response Change Decay" -- each of those slides has ONE <img> in a plain
// .image-card (identical markup/sizing to every other single-image slide,
// no .r-stack) plus a few invisible .fragment markers carrying the next/
// previous image path. This replaced an .r-stack + N overlaid .image-cards
// approach: reveal.js's own ".r-stack > * { margin: auto; }" disables CSS
// Grid's default stretch sizing, and even after overriding that plus adding
// grid-template-columns:1fr, an r-stack slide's image still measured ~3%
// wider (measured directly via getBoundingClientRect, not guessed) than a
// plain .image-card at the same width% -- some remaining interaction with
// grid auto-sizing that wasn't worth chasing further. Swapping one <img>'s
// src on a plain, already-correctly-sized .image-card sidesteps the whole
// question: there is no grid, so there is nothing left to size wrong.
(function () {
  function swapSrc(fragmentEl, attr) {
    var targetId = fragmentEl.getAttribute('data-swap-target');
    var src = fragmentEl.getAttribute(attr);
    if (!targetId || !src) return;
    var img = document.getElementById(targetId);
    if (!img) return;
    img.setAttribute('data-src', src);
    img.setAttribute('src', src);
  }
  function wireUp() {
    if (typeof Reveal === 'undefined' || !Reveal.on) {
      setTimeout(wireUp, 100);
      return;
    }
    Reveal.on('fragmentshown', function (event) {
      swapSrc(event.fragment, 'data-swap-src');
    });
    Reveal.on('fragmenthidden', function (event) {
      swapSrc(event.fragment, 'data-swap-prev');
    });
  }
  wireUp();
})();
