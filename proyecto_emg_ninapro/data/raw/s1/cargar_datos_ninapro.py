import scipy.io
import numpy as np
import os

# Ruta corregida con el nombre de usuario correcto
base_dir = "/home/chigurth/Documentos/Python/DataFrameBitXoma/proyecto_emg_ninapro/data/raw/s1/"
file_path = os.path.join(base_dir, "S1_A1_E1.mat")

# Verificar si el archivo existe
if not os.path.exists(file_path):
    print(f"El archivo no existe en la ruta: {file_path}")
    print("Verificando la existencia del directorio...")
    print(f"¿Existe el directorio? {os.path.exists(base_dir)}")
    
    if os.path.exists(base_dir):
        print("\nArchivos en el directorio:")
        print(os.listdir(base_dir))
    else:
        print(f"El directorio {base_dir} no existe")
else:
    print(f"Archivo encontrado en: {file_path}")
    print(f"Tamaño del archivo: {os.path.getsize(file_path) / (1024*1024):.2f} MB")