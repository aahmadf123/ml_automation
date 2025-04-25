/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["lucide-react"],
  // Fix for 404 errors on static chunks
  output: 'standalone',
  distDir: '.next',
}

module.exports = nextConfig 