#!/bin/bash

# Script para procesar todos los videos y generar el dataset
# Correr una vez despues de clonar el proyecto:
#   bash procesar_datos.sh

cd "$(dirname "$0")"

# Aplicar parche en mediapipe para que funcione en servidores sin GPU (como Lightning AI)
MEDIAPIPE_PATCH=$(python3 -c "import mediapipe; import os; print(os.path.dirname(mediapipe.__file__))")/tasks/python/core/optional_dependencies.py
sed -i 's/except ModuleNotFoundError:/except (ModuleNotFoundError, ImportError, Exception):/' "$MEDIAPIPE_PATCH"

echo "Iniciando procesamiento de videos..."

# Detectar si estamos en un servidor sin pantalla (como Lightning AI)
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "Entorno sin pantalla detectado (servidor). Usando modo CPU."
    __EGL_VENDOR_LIBRARY_FILENAMES='' EGL_PLATFORM=surfaceless python3 modules/preprocess/preprocess.py
else
    echo "Entorno con pantalla detectado. Corriendo normal."
    python3 modules/preprocess/preprocess.py
fi

echo "Procesamiento terminado."
