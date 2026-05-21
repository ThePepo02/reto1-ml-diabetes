# Parte 1 - Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Parte 2 - Cargar datos con todas las variables disponibles
df = pd.read_csv('diabetes.csv')
columnas = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'DiabetesPedigreeFunction', 'Outcome']
df = df[columnas]
print("Primeras filas del dataset mejorado:")
print(df.head())
print(f"Tamaño del dataset: {df.shape}")

# Parte 3 - Normalizar glucosa
media = np.mean(df['Glucose'])
desv  = np.std(df['Glucose'])
df['glucosa_norm'] = (df['Glucose'] - media) / desv
print("\nNormalización:")
print(df[['Glucose', 'glucosa_norm']].head())

# Parte 4 - Scatter plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(df['Glucose'], df['BMI'], c=df['Outcome'], cmap='coolwarm', alpha=0.7)
plt.colorbar(sc, label='Diabetes (0=Sano, 1=Diabético)')
plt.xlabel('Glucosa')
plt.ylabel('IMC (BMI)')
plt.title('Glucosa vs IMC por diagnóstico - Modelo Mejorado')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Parte 5 - Preparar datos con más variables
x = df[['Glucose', 'Age', 'BMI', 'BloodPressure', 'Insulin', 'DiabetesPedigreeFunction']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"\nTotal registros  : {len(df)}")
print(f"Entrenamiento    : {len(X_train)}")
print(f"Prueba           : {len(X_test)}")

# Parte 6 - Entrenar árbol con mayor profundidad
clasificador = DecisionTreeClassifier(max_depth=5, random_state=42)
clasificador.fit(X_train, y_train)

# Parte 7 - Evaluar precisión
predicciones = clasificador.predict(X_test)
precision    = accuracy_score(y_test, predicciones)
print(f"\nPrecisión modelo original  : 70.13%")
print(f"Precisión modelo mejorado  : {precision * 100:.2f}%")

# Parte 8 - Visualizar árbol mejorado
plt.figure(figsize=(24, 12))
plot_tree(clasificador,
          feature_names=['Glucosa', 'Edad', 'IMC', 'Presión Arterial', 'Insulina', 'Pedigree Diabetes'],
          class_names=['Sano', 'Diabético'],
          filled=True,
          rounded=True,
          fontsize=8)
plt.title("Árbol de Decisión Mejorado — Diagnóstico de Diabetes", fontsize=14)
plt.tight_layout()
plt.show()