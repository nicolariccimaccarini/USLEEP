#!/bin/bash
# ======================================================
# Script per scaricare i risultati dal supercomputer Copernico
# Autore: nmaccarini
# ======================================================

# === Parametri ===
REMOTE_USER="nmaccarini"
REMOTE_HOST="copernico.unife.it"
SSH_KEY="$HOME/.ssh/keys_for_hpc"
REMOTE_PATH="/hpc/groups/users-ai/EEG/ML-for-Spindle-Detection-in-EEESWAS/Data/Output/"
LOCAL_PATH="/mnt/c/Users/nicol/OneDrive/Documenti/GitHub/ML-for-Spindle-Detection-in-EEESWAS/Data/Output"

# === Creazione cartella locale se non esiste ===
mkdir -p "$LOCAL_PATH"

echo "🔗 Connessione a $REMOTE_USER@$REMOTE_HOST..."
echo "📦 Copio i dati da $REMOTE_PATH a $LOCAL_PATH"
echo "-----------------------------------------------------"

# === Trasferimento con rsync ===
rsync -avzP -e "ssh -i $SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" \
    "$LOCAL_PATH"

echo "-----------------------------------------------------"
echo "✅ Trasferimento completato!"
echo "I file si trovano in: $LOCAL_PATH"
