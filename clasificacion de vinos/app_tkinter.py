"""Interfaz de escritorio Tkinter para el clasificador MLP de vinos."""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model():
    np.random.seed(42)
    tf.random.set_seed(42)
    wine = load_wine()
    X = wine.data.astype(np.float32)
    y = wine.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

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
    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=200,
        batch_size=16,
        callbacks=[stopper],
        verbose=0,
    )
    predictions = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    return wine, X, y, model, scaler, y_test, predictions


class WineClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de vinos | Red neuronal MLP")
        self.root.geometry("1000x720")
        self.root.minsize(850, 620)
        self.root.configure(padx=18, pady=14)

        self.status = tk.StringVar(value="Entrenando el modelo, espera un momento…")
        ttk.Label(root, text="Clasificación de vinos con una red neuronal", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(root, text="Interfaz de escritorio Tkinter · Dataset Wine Recognition").pack(anchor="w", pady=(2, 12))
        ttk.Label(root, textvariable=self.status).pack(anchor="w", pady=(0, 10))
        self.root.update_idletasks()

        try:
            self.wine, self.X, self.y, self.model, self.scaler, self.y_test, self.predictions = train_model()
        except Exception as exc:
            messagebox.showerror("Error al entrenar", f"No se pudo entrenar el modelo:\n{exc}")
            self.root.destroy()
            return

        self.status.set("Modelo listo.")
        self.build_tabs()

    def build_tabs(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        model_tab = ttk.Frame(notebook, padding=14)
        predict_tab = ttk.Frame(notebook, padding=14)
        notebook.add(model_tab, text="Modelo y resultados")
        notebook.add(predict_tab, text="Probar predicción")

        ttk.Label(model_tab, text="Arquitectura de la MLP", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            model_tab,
            text="13 entradas → Dense(32, ReLU) → Dropout(20%) → Dense(16, ReLU) → Dense(3, Softmax)",
            wraplength=900,
        ).pack(anchor="w", pady=(8, 4))
        ttk.Label(model_tab, text="Optimizador: Adam  |  Pérdida: sparse categorical crossentropy  |  Épocas máximas: 200").pack(anchor="w")
        accuracy = accuracy_score(self.y_test, self.predictions)
        ttk.Label(model_tab, text=f"Exactitud en prueba: {accuracy:.1%}", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(14, 8))

        result_frame = ttk.Frame(model_tab)
        result_frame.pack(fill="both", expand=True, pady=(2, 8))
        class_names = list(self.wine.target_names)
        matrix = confusion_matrix(self.y_test, self.predictions)
        ttk.Label(result_frame, text="Matriz de confusión (filas: real; columnas: predicción)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tree = ttk.Treeview(result_frame, columns=("real", *class_names), show="headings", height=4)
        tree.heading("real", text="Clase real")
        tree.column("real", width=150, anchor="center")
        for name in class_names:
            tree.heading(name, text=name)
            tree.column(name, width=130, anchor="center")
        for name, row in zip(class_names, matrix):
            tree.insert("", "end", values=(name, *row.tolist()))
        tree.pack(anchor="w", pady=(4, 12))

        report = classification_report(
            self.y_test, self.predictions, target_names=class_names, output_dict=True, zero_division=0
        )
        ttk.Label(result_frame, text="Métricas por clase (precisión, recall y F1-score)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        metrics = ttk.Treeview(result_frame, columns=("precision", "recall", "f1-score"), show="headings", height=4)
        for column in ("precision", "recall", "f1-score"):
            metrics.heading(column, text=column.capitalize())
            metrics.column(column, width=150, anchor="center")
        for name in class_names:
            row = report[name]
            metrics.insert("", "end", values=(f"{row['precision']:.3f}", f"{row['recall']:.3f}", f"{row['f1-score']:.3f}"), text=name)
        metrics.pack(anchor="w", pady=4)
        # El nombre de clase se muestra con una primera columna dedicada.
        metrics.configure(show="tree headings")
        metrics.heading("#0", text="Clase")
        metrics.column("#0", width=150, anchor="center")
        for item, name in zip(metrics.get_children(), class_names):
            metrics.item(item, text=name)

        self.build_predict_tab(predict_tab)

    def build_predict_tab(self, parent):
        ttk.Label(parent, text="Clasifica una muestra", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(parent, text="Selecciona una muestra del dataset para cargar sus valores automáticamente, o edita los campos manualmente.", wraplength=900).pack(anchor="w", pady=(4, 10))

        sample_options = [f"Muestra {i + 1} · clase real: {self.wine.target_names[self.y[i]]}" for i in range(len(self.X))]
        self.sample_choice = tk.StringVar(value=sample_options[0])
        selector = ttk.Frame(parent)
        selector.pack(fill="x", pady=(0, 10))
        ttk.Label(selector, text="Muestra de ejemplo:").pack(side="left")
        self.sample_combo = ttk.Combobox(selector, textvariable=self.sample_choice, values=sample_options, state="readonly", width=54)
        self.sample_combo.pack(side="left", padx=8)
        self.sample_combo.bind("<<ComboboxSelected>>", self.load_sample)

        fields = ttk.Frame(parent)
        fields.pack(fill="x")
        self.entries = []
        means = self.X.mean(axis=0)
        for index, (name, value) in enumerate(zip(self.wine.feature_names, self.X[0])):
            row, col = divmod(index, 2)
            cell = ttk.Frame(fields, padding=(0, 3, 14, 3))
            cell.grid(row=row, column=col, sticky="ew")
            ttk.Label(cell, text=name, width=28).pack(side="left")
            entry = ttk.Entry(cell, width=16)
            entry.pack(side="left", padx=(4, 0))
            entry.insert(0, f"{value:.4f}")
            self.entries.append(entry)
        fields.columnconfigure(0, weight=1)
        fields.columnconfigure(1, weight=1)

        ttk.Button(parent, text="Clasificar muestra", command=self.predict,).pack(anchor="w", pady=(14, 8))
        self.prediction_label = ttk.Label(parent, text="", font=("Segoe UI", 13, "bold"))
        self.prediction_label.pack(anchor="w", pady=(2, 8))
        self.probability_labels = []
        for name in self.wine.target_names:
            label = ttk.Label(parent, text=f"{name}: —")
            label.pack(anchor="w", pady=2)
            self.probability_labels.append(label)

    def load_sample(self, _event=None):
        index = self.sample_combo.current()
        for entry, value in zip(self.entries, self.X[index]):
            entry.delete(0, tk.END)
            entry.insert(0, f"{value:.4f}")

    def predict(self):
        try:
            values = np.array([float(entry.get()) for entry in self.entries], dtype=np.float32).reshape(1, -1)
        except ValueError:
            messagebox.showwarning("Dato no válido", "Revisa que los 13 campos contengan números.")
            return
        scaled = self.scaler.transform(values)
        probabilities = self.model.predict(scaled, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        self.prediction_label.configure(text=f"Clase predicha: {self.wine.target_names[predicted_index]}")
        for label, name, probability in zip(self.probability_labels, self.wine.target_names, probabilities):
            label.configure(text=f"{name}: {probability:.1%}")


if __name__ == "__main__":
    root = tk.Tk()
    WineClassifierApp(root)
    if root.winfo_exists():
        root.mainloop()
