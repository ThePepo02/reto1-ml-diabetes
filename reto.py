import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

#Parte 1 filtrar el csv con las columnas que queremos
df = pd.read_csv('diabetes.csv')
columnas = ['Glucose', 'BMI', 'Age', 'Outcome']
df = df[columnas]
print("Primeras filas:")
print(df.head())
print(f"Tamaño del dataset: {df.shape}")

#Parte 2 calculo de la medio y la desviación normalizando los datos en la misma escala
media = np.mean(df['Glucose'])
desv = np.std(df['Glucose'])
df['glucosa_norm'] = (df['Glucose'] - media) / desv
print("\nNormalización")
print(df[['Glucose', 'glucosa_norm']].head())


#Parte 3 Grafico de dispercion Matplotlib
plt.figure(figsize=(10, 6))
sc = plt.scatter(df['Glucose'], df['BMI'], c=df['Outcome'], cmap='coolwarm', alpha=0.7)
plt.colorbar(sc, label='Diabetes (0=Sano, 1=Diabético)')
plt.xlabel('Glucosa')
plt.ylabel('IMC (BMI)')
plt.title('Glucosa vs IMC por diagnóstico')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


#Parte 4  Entrenamos a la Sklearn para saber que datos tratamos y como estan repartidos
x = df[['Glucose', 'Age', 'BMI']]
y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Total registros : {len(df)}")
print(f"Entrenamiento : {len(X_train)}")
print(f"Prueba : {len(X_test)}")

clasificador = DecisionTreeClassifier(max_depth=4, random_state=42)
clasificador.fit(X_train, Y_train)