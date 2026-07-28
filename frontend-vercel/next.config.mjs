import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Pin the workspace root to this directory. Without this, Next.js/Turbopack
  // can get confused by an unrelated package-lock.json higher up in the
  // filesystem (this app lives in a subdirectory of a larger, messy home
  // directory) and infer the wrong project root.
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
