"""Point d'entrée de ML Forge.

Lance l'application de bureau.
"""

import sys
from pathlib import Path

# Ajouter le répertoire courant au chemin Python
sys.path.insert(0, str(Path(__file__).parent))

from ml_forge.gui.app import main

if __name__ == "__main__":
    main()
