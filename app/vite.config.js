import { defineConfig } from "vite";

// Relative base so the same build works from the Capacitor WebView,
// from the Python bridge's static server, and from GitHub Pages.
export default defineConfig({
  base: "./",
  build: { outDir: "dist", chunkSizeWarningLimit: 900 },
});
