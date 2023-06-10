from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import date
import numpy as np
import pandas as pd

# Datos de entrada
dataset1 = pd.read_csv('./DISASTERS/DISASTER1.csv')
dataset2 = pd.read_csv('./DISASTERS/DISASTER2.csv')
dataset = pd.concat([dataset1, dataset2])
dataset = dataset.drop_duplicates()

pais = "Colombia"
catastrofe = "Earthquake"

def prediction(pais: str, catastrofe: str):
    # Seleccionamos la columna de los datos de entrada
    datos_pais = dataset["Country"].values
    datos_del_pais = dataset.loc[dataset["Country"] == pais]
    catastrofe = dataset["Disaster Type"].values
    # print("Datos del pais: ", datos_del_pais)
    # datos_catastrofe = datos_del_pais.loc[dataset["Disaster Type"] == "Earthquake"]
    datos_catastrofe = datos_del_pais.merge(
        dataset[dataset["Disaster Type"] == catastrofe], on="Disaster Type")
    # print("Datos de la catastrofe: \n", datos_catastrofe)
    datos_catastrofe.dropna(axis=0, subset=["Dis Mag Value_x"], inplace=True)
    # print("Datos de la catastrofe: \n", datos_catastrofe)
    year = np.reshape(datos_catastrofe["Year_x"].values, (-1, 1))
    magnitud = datos_catastrofe["Dis Mag Value_x"].values
    magnitud = np.reshape(magnitud, (-1, 1))
    # print("Magnitud: ", magnitud)
    # print("Year: ", year)

    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(year, magnitud)
    # predecir los datos de entrada de prueba (datos de entrada) usando el modelo entrenado (se encuentra en la variable model)
    prediction = linear_regressor.predict(year)
    # Se grafican los datos de entrada y la linea de regresion
    prediction = prediction.sum()/prediction.size

    # Datos de ejemplo
    objetivo = np.array([1 if i % 2 == 0 else 0 for i in range(
        0, datos_catastrofe.shape[0])])  # Variable objetivo binaria (0 o 1)

    # Crear una matriz de características combinando magnitudes y años
    caracteristicas = np.column_stack((magnitud, year))

    # Crear y entrenar el modelo de regresión logística
    modelo = LogisticRegression()
    modelo.fit(caracteristicas, objetivo)
    nueva_magnitud = prediction
    nuevo_anio = date.today().year
    nuevo_ejemplo = np.array([[nueva_magnitud, nuevo_anio]])
    probabilidad = modelo.predict_proba(nuevo_ejemplo)[:, 1]
    print(f"Probabilidad de que ocurra un desastre de: {prediction} es de: {float(probabilidad)*100}%")
    return {
        "probabilidad": float(probabilidad)*100,
        "magnitud": prediction,
    }
    # print(prediction.sum()/prediction.size)
    # plt.figure()
    # plt.scatter(magnitud, year)
    # plt.plot(magnitud, year, color='red')
    # plt.show()
