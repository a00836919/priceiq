import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export → SPA HTML/JS sirviendo desde CDN (Azure Static Web Apps Free).
  output: "export",
  // El optimizador de imágenes de Next.js requiere servidor Node.js; lo desactivamos
  // para que las <Image /> sirvan como <img> normales desde el CDN.
  images: { unoptimized: true },
  // Genera carpetas con index.html en lugar de archivos sueltos (mejor compat. con CDNs)
  trailingSlash: true,
};

export default nextConfig;
