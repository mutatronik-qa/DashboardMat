# 🦾 DashboardMat - NinaPro EMG Data Analysis Platform

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![NinaPro DB1](https://img.shields.io/badge/NinaPro-DB1-orange?style=for-the-badge)](https://ninapro.hevs.ch/)

> Plataforma de análisis de datos EMG para investigación en Prótesis Robóticas - Desplegada en Streamlit Cloud

## 📋 Descripción del Proyecto

**DashboardMat** es una herramienta integral para el análisis de archivos `.mat` de la base de datos [NinaPro](https://ninapro.hevs.ch/), diseñada específicamente para investigación en prótesis robóticas y desarrollo de sistemas de reconocimiento de gestos mediante señales electromiográficas (EMG).

### 🎯 Objetivos Principales

1. **Análisis Exploratorio**: Visualización interactiva de señales EMG y datos del guante sensor
2. **Procesamiento de Datos**: Pipeline completo de preprocesamiento y extracción de características
3. **Machine Learning**: Entrenamiento de modelos para reconocimiento de movimientos
4. **Replicación de Sensores**: Base para desarrollar réplicas de los sensores y guantes del estudio NinaPro

---

## 🏗️ Arquitectura del Proyecto

```
DashboardMat/
├── 📁 Dashboard/                    # Aplicación Streamlit interactiva
│   ├── app.py                      # Dashboard principal de visualización
│   └── requirements.txt            # Dependencias del dashboard
│
├── 📁 proyecto_emg_ninapro/        # Paquete de procesamiento EMG
│   ├── data/
│   │   └── raw/                    # Archivos .mat crudos (S1_A1_E1.mat, etc.)
│   └── src/
│       ├── data_loader.py          # Carga de datos NinaPro
│       ├── preprocessing.py        # Preprocesamiento de señales
│       ├── feature_extraction.py  # Extracción de características
│       └── models.py               # Modelos de ML (RF, SVM)
│
└── 📁 CascadeProjects/
    └── windsurf-project/           # Utilidades genéricas .mat
        └── mat_processor/
            ├── io.py               # Lectura de archivos .mat
            ├── analysis.py         # Análisis estadístico
            ├── visualization.py    # Visualizaciones matplotlib
            └── cli.py             # Interfaz de línea de comandos
```

---

## 🔬 Base de Datos NinaPro

### Acerca de NinaPro

La base de datos [NinaPro](https://ninapro.hevs.ch/) (Non-Invasive Adaptive Prosthetics) es un recurso público fundamental para la investigación en interfaces humano-máquina y prótesis mioeléctricas. Fue desarrollada por el grupo de investigación de la **Haute École Spécialisée de Suisse Occidentale (HES-SO)** en Suiza.

### NinaPro DB1 - Contenido

El proyecto está optimizado para **NinaPro DB1**, que contiene:

| Característica | Descripción |
|----------------|-------------|
| **Sujetos** | 27 sujetos saludables |
| **Electrodos EMG** | 10 electrodos de superficie (8 canales EMG + 2 de referencia) |
| **Guante de datos** | 22 sensores de posición de dedos (Deman Robotics) |
| **Ejercicios** | 3 ejercicios por sujeto |
| **Frecuencia de muestreo** | 100 Hz |

### Ejercicios DB1

| Ejercicio | Descripción | Movimientos |
|-----------|-------------|-------------|
| **E1** | Flexiones básicas de dedos | 12 movimientos + reposo |
| **E2** | Fuerza isométrica/isotónica | 17 movimientos |
| **E3** | Patrones de agarre | 23 movimientos |

### Estructura de Archivos .mat

Cada archivo `.mat` contiene las siguientes variables:

| Variable | Descripción | Dimensión |
|----------|-------------|-----------|
| `emg` | Señales EMG crudas | (n_muestras, 10) |
| `stimulus` | Etiquetas de movimiento | (n_muestras,) |
| `repetition` | Número de repetición | (n_muestras,) |
| `restimulus` | Etiqueta de movimiento real | (n_muestras,) |
| `glove` | Datos del guante sensor | (n_muestras, 22) |
| `subject` | ID del sujeto | (1, 1) |

---

## 🚀 Despliegue en Streamlit Cloud

### Requisitos Previos

```txt
# Dashboard/requirements.txt
streamlit>=1.30.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
plotly>=5.18.0
h5py>=3.9.0
mlflow>=2.10.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### Pasos de Despliegue

1. **Preparar el Repositorio**
   ```bash
   # Estructura requerida en GitHub
   DashboardMat/
   ├── Dashboard/
   │   ├── app.py
   │   └── requirements.txt
   └── (otros archivos)
   ```

2. **Configurar Streamlit Cloud**
   - Conectar tu repositorio GitHub a [Streamlit Cloud](https://streamlit.io/cloud)
   - Seleccionar el branch principal (main/master)
   - Establecer la ruta del archivo: `Dashboard/app.py`
   - Streamlit detectará automáticamente las dependencias

3. **Configuración Adicional (opcional)**
   - Crear `.streamlit/config.toml` para configuraciones personalizadas:
   ```toml
   [server]
   port = 8501
   headless = true
   
   [theme]
   primaryColor = "#FF4B4B"
   backgroundColor = "#0E1117"
   ```

### Uso del Dashboard

1. **Cargar Archivos .mat**
   - Desde ruta local: Seleccionar archivo del directorio
   - Subir archivo: Arrastrar y soltar archivo `.mat`

2. **Explorar Señales EMG**
   - Visualización de canales EMG individuales
   - Timeline de estímulos/movimientos
   - Distribución estadísticas por canal
   - Matriz de correlación

3. **Análisis de Datos**
   - Descarga de datos en CSV
   - Exploración de estructura cruda del archivo

---

## 💻 Uso Local

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/DashboardMat.git
cd DashboardMat

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r Dashboard/requirements.txt
pip install -r proyecto_emg_ninapro/requirements.txt
```

### Ejecutar el Dashboard

```bash
cd Dashboard
streamlit run app.py
```

### Uso del Paquete de Procesamiento EMG

```python
from proyecto_emg_ninapro.src.data_loader import cargar_datos_ninapro
from proyecto_emg_ninapro.src.preprocessing import preprocess_pipeline
from proyecto_emg_ninapro.src.feature_extraction import extract_all_features

# Cargar datos
data = cargar_datos_ninapro('ruta/al/archivo.mat')

# Preprocesar
processed = preprocess_pipeline(data, fs=100.0)

# Extraer características
features = extract_all_features(processed['emg'])
```

---

## 🔧 Módulos del Proyecto

### 1. Dashboard (Streamlit)

Aplicación interactiva para visualización de datos:
- Explorador de archivos .mat
- Visualización de señales EMG
- Timeline de movimientos
- Análisis estadístico
- Correlación entre canales
- Densidad espectral de potencia

### 2. Preprocesamiento (`preprocessing.py`)

- Filtro paso banda (20-450 Hz)
- Filtro notch (50/60 Hz)
- Normalización (Standard, MinMax, MVC)
- Segmentación de señales

### 3. Extracción de Características (`feature_extraction.py`)

**Dominio Temporal:**
- MAV (Mean Absolute Value)
- RMS (Root Mean Square)
- WL (Waveform Length)
- VAR (Variance)
- SSI (Simple Square Integral)
- ZC (Zero Crossing)
- SSC (Slope Sign Change)
- Skewness & Kurtosis

**Dominio Frecuencial:**
- MNF (Mean Frequency)
- MDF (Median Frequency)
- PKF (Peak Frequency)
- PSD (Power Spectral Density)

### 4. Modelos de Machine Learning (`models.py`)

- Random Forest Classifier
- Support Vector Machine (SVM)
- Validación cruzada
- Búsqueda de hiperparámetros
- Importancia de características

---

## 📚 Comparación con Repositorios de Referencia

### Este Proyecto vs [Zeng-Jia/Ninapro-dataset-processing](https://github.com/Zeng-Jia/Ninapro-dataset-processing)

| Característica | DashboardMat | Zeng-Jia/Ninapro-dataset-processing |
|----------------|--------------|-------------------------------------|
| **Interfaz** | Streamlit Web (navegador) | Scripts Python / Jupyter |
| **Despliegue** | Streamlit Cloud (público) | Local |
| **Visualización** | Plotly interactivo | Matplotlib estático |
| **Machine Learning** | sklearn integrado | Keras/TensorFlow |
| **API REST** | ❌ No | Posible extensión |
| **Procesamiento por lotes** | Limitado | Completo |

### Similitudes

- Ambos procesan archivos .mat de NinaPro
- Extracción de características EMG estándar
- Soporte para DB1 y ejercicios (E1, E2, E3)
- Código abierto y reproducible

---

## 📎 Recursos Externos

### Base de Datos

- 🌐 **NinaPro Official**: [https://ninapro.hevs.ch/](https://ninapro.hevs.ch/)
- 📊 **Kaggle Dataset**: [NinaPro DB1 Full Dataset](https://www.kaggle.com/datasets/mansibmursalin/ninapro-db1-full-dataset)

### Repositorios de Referencia

- 🔬 [Zeng-Jia/Ninapro-dataset-processing](https://github.com/Zeng-Jia/Ninapro-dataset-processing)
- 📄 [Paper Original NinaPro](https://doi.org/10.1109/TNSRE.2014.2304950)

### Documentación Técnica

- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [h5py Documentation](https://docs.h5py.org/)

---

## 🎓 Aplicaciones y Casos de Uso

1. **Investigación Académica**
   - Análisis de señales EMG para estudios de movimiento
   - Desarrollo de algoritmos de clasificación

2. **Desarrollo de Prótesis**
   - Entrenamiento de modelos para control mioeléctrico
   - Replicación de sensores y guantes del estudio

3. **Educación**
   - Tutoriales de procesamiento de señales biomédicas
   - Ejemplos de machine learning aplicado

4. **Prototipado**
   - Experimentación rápida con datos reales
   - Validación de hipótesis de investigación

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request.

---

## 📊 Estado del Proyecto

- ✅ Dashboard Streamlit funcional
- ✅ Carga de archivos .mat (v5 y v7.3)
- ✅ Visualización de señales EMG
- ✅ Extracción de características
- 🔄 Integración de modelos ML
- 📅 Despliegue en Streamlit Cloud

---

*Made with ❤️ for prosthetics research*
