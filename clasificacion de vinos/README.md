# Clasificación de vinos con una red neuronal MLP

## Problema

El objetivo es predecir a cuál de tres clases pertenece una muestra de vino a partir de 13 características químicas. Se usa el dataset **Wine Recognition** que viene incluido en scikit-learn: contiene 178 muestras y no requiere descargar un archivo aparte.

## Enfoque

El programa `clasificacion_vinos_mlp.py` implementa una red neuronal multicapa con TensorFlow/Keras. Divide los datos en entrenamiento y prueba conservando la proporción de las clases, normaliza las características con `StandardScaler` y entrena una MLP con dos capas ocultas. La salida `softmax` devuelve probabilidades para las tres clases. Se usa `sparse_categorical_crossentropy` porque las etiquetas son enteros (0, 1 y 2), y `EarlyStopping` para detener el entrenamiento si la pérdida de validación deja de mejorar.

La evaluación final se hace sobre el conjunto de prueba y muestra exactitud, precisión, recall, F1-score y matriz de confusión. La partición de prueba no se usa para ajustar el escalador ni para entrenar la red.

## Requisitos

- Python 3.10 o superior
- TensorFlow
- scikit-learn
- NumPy
- Tkinter (incluido normalmente con Python)

Instala las dependencias con:

```bash
python -m pip install tensorflow scikit-learn numpy
```

## Interfaz de escritorio Tkinter

Instala las dependencias y abre la interfaz con:

```bash
python -m pip install -r requirements.txt
python app_tkinter.py
```

Se abrirá una ventana nativa de escritorio con pestañas para ver la arquitectura y evaluación del modelo y para probar predicciones. Puedes cargar una muestra del dataset o editar manualmente las 13 características. El entrenamiento se realiza al abrir la ventana. Como el dataset es pequeño, el resultado puede variar ligeramente entre versiones o entornos.

También puedes ejecutar `python clasificacion_vinos_mlp.py` para obtener las métricas en la terminal.

## Elementos para presentar

1. **Problema:** clasificación supervisada multiclase.
2. **Entrada:** 13 mediciones químicas por muestra.
3. **Salida:** una de las tres clases de vino.
4. **Preprocesamiento:** división estratificada y estandarización de variables.
5. **Modelo:** MLP con capas densas ReLU, dropout y capa final softmax.
6. **Evaluación:** exactitud, precisión, recall, F1-score y matriz de confusión.

En la conclusión del informe, interpreta las métricas que imprima tu ejecución y comenta qué clases se confunden según la matriz; evita afirmar un resultado antes de ejecutar el programa.
