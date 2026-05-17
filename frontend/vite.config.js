// import { defineConfig } from 'vite'
// import react from '@vitejs/plugin-react'

// export default defineConfig({
//   plugins: [react()],
//   server: {
//     proxy: {
//       '/api': {
//         target: 'http://localhost:8000',
//         changeOrigin: true,
//         rewrite: (path) => path.replace(/^\/api/, ''),
//         timeout: 120000,
//         configure: (proxy) => {
//           proxy.on('proxyReq', (proxyReq) => {
//             proxyReq.setTimeout(120000);
//           });
//         },
//       },
//     },
//   },
// })

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        timeout: 120000,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setTimeout(120000);
          });
        },
      },
      // Add this for /chats
      '/chats': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        timeout: 120000,
      },
    },
  },
})