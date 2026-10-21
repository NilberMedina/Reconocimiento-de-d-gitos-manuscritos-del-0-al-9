"""Clasificación de vinos con una red neuronal multicapa (MLP).

Dataset: Wine Recognition, incluido en scikit-learn.
Framework de red neuronal: TensorFlow/Keras.
"""

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # Fijamos las semillas para que la partición y el entrenamiento sean
    # razonablemente reproducibles.
    np.random.seed(42)
    tf.random.set_seed(42)

    # El dataset tiene 178 muestras, 13 características químicas y 3 clases.
    wine = load_wine()
    X = wine.data.astype(np.float32)
    y = wine.target.astype(np.int64)

    # Separamos antes de ajustar el escalador para evitar fuga de información.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # MLP: 13 entradas, dos capas ocultas y 3 salidas (una por clase).
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(len(wine.target_names), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=200,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=0,
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print("Dataset: Wine Recognition")
    print(f"Muestras: {len(X)} | Características: {X.shape[1]} | Clases: {len(wine.target_names)}")
    print(f"Exactitud en prueba: {accuracy_score(y_test, predictions):.4f}")
    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=wine.target_names,
            zero_division=0,
        )
    )
    print("Matriz de confusión (filas = clase real, columnas = predicción):")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
