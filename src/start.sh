#!/bin/bash
set -e

# --- KONFIGURASJON ---
# Bytt ut denne med din GitHub URL når repoet er public
REPO_URL="https://github.com/torsteinelv/qwen3-tts-norwegian.git"
BRANCH="main"
CODE_DIR="/workspace/github_code" # Vi cloner hit først for å unngå konflikter

echo "=================================================="
echo "   🚀 QWEN3-TTS BOOTSTRAPPER (Auto-Update)        "
echo "=================================================="

# 1. Sjekk internett/git tilgang
echo "📡 Kobler til GitHub..."

if [ -d "$CODE_DIR/.git" ]; then
    echo "🔄 Repo funnet lokalt. Henter oppdateringer..."
    cd "$CODE_DIR"
    git fetch origin
    git reset --hard origin/$BRANCH
    git pull origin $BRANCH
else
    echo "⬇️ Første oppstart. Cloner repository..."
    rm -rf "$CODE_DIR"
    git clone -b $BRANCH "$REPO_URL" "$CODE_DIR"
fi

# 2. Overskriv koden i containeren med den ferske koden
# Vi kopierer innholdet fra src i repoet til /workspace/src der containeren forventer det
echo "📦 Installerer nyeste kode til /workspace/src..."
cp -r "$CODE_DIR/src/"* /workspace/src/

# 3. Sørg for at alt er kjørbart
chmod +x /workspace/src/*.sh
chmod +x /workspace/src/*.py

echo "✅ Koden er oppdatert! Starter hovedprosessen..."
echo "--------------------------------------------------"

# 4. Kjør entrypoint (med evt. argumenter du sendte til docker run)
# 'exec' erstatter denne prosessen med entrypoint, slik at signaler (Ctrl+C) virker riktig
exec /workspace/src/entrypoint.sh "$@"
