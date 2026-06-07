#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="/mnt/c/Users/aaron/AppData/Roaming/Elgato/StreamDeck/Plugins/com.topoconfidence.daemon-monitor.sdPlugin"

cd "$SCRIPT_DIR"

echo "Building..."
npx esbuild src/plugin.ts \
  --bundle \
  --platform=node \
  --target=node20 \
  --format=esm \
  --outfile=dist/bin/plugin.js \
  --banner:js="import{createRequire}from'module';const require=createRequire(import.meta.url);"

echo "Deploying to $PLUGIN_DIR ..."
mkdir -p "$PLUGIN_DIR/bin" "$PLUGIN_DIR/ui" "$PLUGIN_DIR/imgs/actions/daemon-monitor"

cp dist/bin/plugin.js "$PLUGIN_DIR/bin/"
cp manifest.json "$PLUGIN_DIR/"
cp ui/*.html "$PLUGIN_DIR/ui/" 2>/dev/null || true
cp ui/sdpi-components.js "$PLUGIN_DIR/ui/"
cp imgs/actions/daemon-monitor/* "$PLUGIN_DIR/imgs/actions/daemon-monitor/" 2>/dev/null || true
cp imgs/category*.png "$PLUGIN_DIR/imgs/" 2>/dev/null || true

echo "Deployed. Restart Stream Deck app to pick up changes."
