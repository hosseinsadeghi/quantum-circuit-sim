const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8001';

// Proxy /api/* to Python FastAPI backend.
// Mounted at root (not '/api') so Express does not strip the prefix from req.url
// before http-proxy-middleware forwards it. pathFilter handles the routing.
app.use(
  createProxyMiddleware({
    target: PYTHON_API_URL,
    changeOrigin: true,
    pathFilter: '/api/**',
    on: {
      error: (err, req, res) => {
        console.error('Proxy error:', err.message);
        res.status(502).json({ error: 'Backend unavailable', detail: err.message });
      },
    },
  })
);

// Serve React frontend static files
const distDir = path.join(__dirname, 'frontend', 'dist');
app.use(express.static(distDir));

// SPA fallback — all non-API routes serve index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(distDir, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Quantum Explorer running at http://localhost:${PORT}`);
  console.log(`Proxying /api → ${PYTHON_API_URL}`);
});
