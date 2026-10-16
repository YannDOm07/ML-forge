#!/bin/sh
# Point d'entrée de l'image ML Forge.
#   docker run ... mlforge <commande forge>   -> python -m forge <commande>
#   docker run -i ... mlforge mcp              -> serveur MCP sur stdio (pour opencode)
#   docker run ... mlforge test                -> suite de tests complète
set -e

if [ "$1" = "test" ]; then
    shift
    exec python -m pytest "$@"
fi

# Première utilisation du volume : configuration et racine des données dans /data.
if [ ! -f "$MLFORGE_CONFIG" ]; then
    python -m forge init --root "$MLFORGE_ROOT" >&2
fi

if [ "$1" = "mcp" ]; then
    exec python -m forge.mcp_server
fi

exec python -m forge "$@"
