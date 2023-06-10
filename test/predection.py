import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos desde un archivo CSV
dataset = pd.read_csv('./DISASTERS/DISASTER1.csv')

# Seleccionar las características relevantes y la variable objetivo
features = dataset[['Year', 'Disaster Type', 'Disaster Subtype', 'Region', 'Total Deaths', 'Total Damages']]
target = dataset['Catastrophe']

# Convertir variables categóricas en variables numéricas utilizando one-hot encoding
features = pd.get_dummies(features)

# Dividir el conjunto de datos en datos de entrenamiento y datos de prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Crear el modelo de clasificación RandomForestClassifier
model = RandomForestClassifier()

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)
