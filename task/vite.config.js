import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig(({ mode }) => {
  // Production builds are still mode-specific
  if (mode === 'binary') {
    return {
      base: './',
      build: {
        outDir: 'dist-binary',
        rollupOptions: { input: resolve(__dirname, 'index-binary.html') },
      },
    };
  }
  if (mode === 'continuous') {
    return {
      base: './',
      build: {
        outDir: 'dist-continuous',
        rollupOptions: { input: resolve(__dirname, 'index-continuous.html') },
      },
    };
  }

  // Dev mode: serve everything, open the launcher page
  return {
    base: '/',
    server: { open: '/index-dev.html' },
    build: {
      // Default build = continuous
      outDir: 'dist-continuous',
      rollupOptions: { input: resolve(__dirname, 'index-continuous.html') },
    },
  };
});
