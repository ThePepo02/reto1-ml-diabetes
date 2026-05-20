
# Parte 1: Importar las librrias necesarias para el análisis de datos, visualización y modelado.
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

#Parte 2 filtrar el csv con las columnas que queremos
df = pd.read_csv('diabetes.csv')
columnas = ['Glucose', 'BMI', 'Age', 'Outcome']
df = df[columnas]
print("Primeras filas:")
print(df.head())
print(f"Tamaño del dataset: {df.shape}")

#Parte 3 calculo de la medio y la desviación normalizando los datos en la misma escala
media = np.mean(df['Glucose'])
desv = np.std(df['Glucose'])
df['glucosa_norm'] = (df['Glucose'] - media) / desv
print("\nNormalización")
print(df[['Glucose', 'glucosa_norm']].head())


#Parte 4 Grafico de dispercion Matplotlib
plt.figure(figsize=(10, 6))
sc = plt.scatter(df['Glucose'], df['BMI'], c=df['Outcome'], cmap='coolwarm', alpha=0.7)
plt.colorbar(sc, label='Diabetes (0=Sano, 1=Diabético)')
plt.xlabel('Glucosa')
plt.ylabel('IMC (BMI)')
plt.title('Glucosa vs IMC por diagnóstico')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


#Parte 5  Entrenamos a la Sklearn para saber que datos tratamos y como estan repartidos
x = df[['Glucose', 'Age', 'BMI']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Total registros : {len(df)}")
print(f"Entrenamiento : {len(X_train)}")
print(f"Prueba : {len(X_test)}")

clasificador = DecisionTreeClassifier(max_depth=4, random_state=42)
clasificador.fit(X_train, y_train)



# Parte 6 medir la precisión del modelo
predicciones = clasificador.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print(f"\nPrecisión del modelo: {precision * 100:.2f}%")


# Parte 7 - Visualizar el árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(clasificador, feature_names=['Glucosa', 'Edad', 'IMC'], 
          class_names=['Sano', 'Diabetico'],
          filled=True,
          rounded=True,
          fontsize=9)
plt.title("Árbol de Decisión - Diagnóstico de Diabetes", fontsize=14)
plt.tight_layout()
plt.show()
