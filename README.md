# IATraductorSe-as

Sistema que detecta gestos de lengua de señas colombiana mediante una cámara y predice en pantalla qué seña se está realizando. Utiliza MediaPipe para extraer landmarks de las manos y un modelo LSTM Bidireccional para clasificar los gestos.

Actualmente soporta **33 gestos de cortesía** (saludos, despedidas, expresiones de agradecimiento, entre otros).

---

## Estructura del proyecto

```
IATraductorSeñas/
├── data/                        # Videos originales organizados por categoría
│   └── Courtesy/
│       └── <gesto>/
│           └── *.avi
├── dataset/                     # Archivos .npy generados + split.csv (no incluido en git)
├── graphics/                    # Gráficas generadas durante el entrenamiento
│   ├── training_metrics.png     # Curvas de loss y accuracy por época
│   └── f1_por_clase.png         # F1-score por gesto
├── modules/
│   ├── preprocess/
│   │   ├── pipeline.py          # Extracción y normalización de landmarks con MediaPipe
│   │   ├── preprocess.py        # Procesa los videos y guarda los .npy
│   │   └── split_dataset.py     # Divide el dataset en train/val/test
│   ├── dataloader.py            # Dataset y DataLoader de PyTorch
│   ├── model.py                 # Arquitectura del modelo LSTM Bidireccional
│   ├── train.py                 # Entrenamiento, evaluación y generación de gráficas
│   ├── evaluar.py               # Evaluación de accuracy por gesto
│   ├── inferencia.py            # Demo con Gradio (muestra landmarks en video)
│   ├── inferencia2.py           # Demo con Gradio (interfaz con botón Traducir)
│   └── inferenciaAPI.py         # API REST con FastAPI (endpoint /predict)
├── test_model/
│   └── hand_landmarker.task     # Modelo preentrenado de MediaPipe
├── procesar_datos.sh            # Script para procesar los videos
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python 3.10+
- pip

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Cómo ejecutar

### 1. Procesar los videos

Extrae los landmarks de cada video y genera los archivos `.npy` normalizados:

```bash
bash procesar_datos.sh
```

### 2. Dividir el dataset

Divide los archivos en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%):

```bash
python3 modules/preprocess/split_dataset.py
```

### 3. Entrenar el modelo

```bash
python3 modules/train.py
```

El modelo se guarda en `modelo_entrenado.pth` cuando mejora la accuracy de validación. Al finalizar, genera automáticamente en la carpeta `graphics/`:
- `training_metrics.png` — curvas de loss y accuracy (train/val/test por época)
- `f1_por_clase.png` — F1-score individual por gesto

También imprime en consola la precisión, recall y F1 por clase sobre el conjunto de prueba.

### 4. Evaluar por gesto

```bash
python3 modules/evaluar.py
```

Muestra la accuracy individual de cada uno de los 33 gestos sobre el conjunto de prueba.

### 5. Demo en tiempo real

**Opción A — Gradio con visualización de landmarks:**

```bash
python3 modules/inferencia.py
```

**Opción B — Aplicación web en tiempo real + API REST:**

Para utilizar la versión web en tiempo real, es necesario ejecutar tanto el backend de inferencia como el servidor HTTP encargado de servir el frontend.

1. Ejecutar la API REST:

cd modules

```bash
cd modules
uvicorn modules.inferenciaAPI:app --host 0.0.0.0 --port 8000
```

2. Ejecutar el frontend:
3. 
```bash
python -m http.server 8081
```

Posteriormente, se puede acceder a la aplicación web desde el navegador mediante el puerto `8081`. Esta versión captura automáticamente fragmentos cortos de video utilizando la cámara del dispositivo y los envía periódicamente a la API para realizar inferencias en tiempo casi real.

### 6. API REST

La API expone el endpoint `POST /predict`, el cual recibe un archivo de video y retorna un JSON con la predicción realizada por el modelo:

```json
{
  "prediccion": "gracias",
  "confianza": 0.87,
  "frames_totales": 45,
  "frames_con_mano": 42
}
```

---


## Modelo preentrenado

El archivo `modelo_entrenado.pth` contiene los pesos del modelo con mejor accuracy de validación. Se puede usar directamente con cualquiera de los scripts de inferencia sin necesidad de reentrenar.

---

## Declaración de uso de IA

Durante el desarrollo de este proyecto se utilizaron las siguientes herramientas de inteligencia artificial:

**GitHub Copilot**
- Propósito: asistencia en la generación de código.
- Partes del proyecto: escritura de funciones, estructura de clases y fragmentos de código repetitivo.

**Claude (Anthropic)**
- Propósito: soporte en pruebas del modelo, identificación de errores y mejora iterativa del sistema.
- Partes del proyecto: depuración del pipeline de datos, diagnóstico de problemas de generalización del modelo (overfitting, domain shift), ajuste de hiperparámetros, y corrección de errores en el código de inferencia.
