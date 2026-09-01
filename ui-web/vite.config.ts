import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          ui: ["@radix-ui/react-dialog", "motion"],
          data: ["@tanstack/react-query", "axios"],
          markdown: ["markdown-it", "dompurify"],
        },
      },
    },
  },
})
