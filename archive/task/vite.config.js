import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig(({ mode }) => {
  // Each mode gets its own dev port so `npm run dev:continuous` and
  // `npm run dev:binary` can run simultaneously in two terminals without a
  // port collision, each auto-opening its own task.
  if (mode === 'binary') {
    return {
      base: './',
      server: { port: 5174, open: '/index-binary.html' },
      build: {
        outDir: 'dist-binary',
        rollupOptions: { input: resolve(__dirname, 'index-binary.html') },
      },
    };
  }
  if (mode === 'continuous') {
    return {
      base: './',
      server: { port: 5173, open: '/index-continuous.html' },
      build: {
        outDir: 'dist-continuous',
        rollupOptions: { input: resolve(__dirname, 'index-continuous.html') },
      },
    };
  }

  // Dev mode (no --mode flag): serve everything, open the continuous task
  // by default (matches the build-mode default below; use
  // `npm run dev:binary` or navigate to /index-binary.html directly for
  // the binary task).
  return {
    base: '/',
    server: { open: '/index-continuous.html' },
    build: {
      // Default build = continuous
      outDir: 'dist-continuous',
      rollupOptions: { input: resolve(__dirname, 'index-continuous.html') },
    },
  };
});
