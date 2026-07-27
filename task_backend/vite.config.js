import { defineConfig } from 'vite';
import { resolve } from 'path';

// GitHub Pages serves this project as a "project page":
// https://<user>.github.io/evidence_integration/ -- everything (JS/CSS
// asset URLs included) needs to resolve under that /evidence_integration/
// prefix, hence `base` below for the real production build. Local dev
// (`npm run dev*`) doesn't need that prefix at all -- it's served from
// the site root regardless of what the eventual deploy path is.
const GITHUB_PAGES_BASE = '/evidence_integration/';

export default defineConfig(({ mode }) => {
  // Each mode gets its own dev port so `npm run dev:numbers` and
  // `npm run dev:colors` can run simultaneously in two terminals without a
  // port collision, each auto-opening its own task. Also usable for a
  // standalone single-task production build (`npm run build:numbers` /
  // `build:colors`) if ever needed, but the REAL deploy path is the
  // combined build below (no --mode), which is what the GitHub Actions
  // workflow actually runs.
  if (mode === 'colors') {
    return {
      base: './',
      server: { port: 5174, open: '/index-colors.html' },
      build: {
        outDir: 'dist-colors',
        rollupOptions: { input: resolve(__dirname, 'index-colors.html') },
      },
    };
  }
  if (mode === 'numbers') {
    return {
      base: './',
      server: { port: 5173, open: '/index-numbers.html' },
      build: {
        outDir: 'dist-numbers',
        rollupOptions: { input: resolve(__dirname, 'index-numbers.html') },
      },
    };
  }

  // No --mode: this is BOTH the default dev server (serves everything,
  // opens the numbers task) AND the real production build target (`npm
  // run build`) -- one combined `dist/` with both index-numbers.html and
  // index-colors.html, deployed as a single GitHub Pages site. One
  // deployment, two pages, per that decision.
  return {
    base: process.env.NODE_ENV === 'production' ? GITHUB_PAGES_BASE : '/',
    server: { open: '/index-numbers.html' },
    build: {
      outDir: 'dist',
      rollupOptions: {
        input: {
          numbers: resolve(__dirname, 'index-numbers.html'),
          colors:  resolve(__dirname, 'index-colors.html'),
        },
      },
    },
  };
});
