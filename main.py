from fastapi import FastAPI
import pandas as pd
import numpy as np
from prediccion import prediction

app = FastAPI()

# Al llamar a la funcion app.get('/') se ejecuta la funcion root, seria como ponerle un sobrenombre
# Se lee un archivo CSV llamado "DISASTER1.csv" utilizando pd.read_csv() de pandas y se guarda en la variable dataset.
dataset = pd.read_csv('./DISASTERS/DISASTER1.csv')

# Se define una ruta principal ("/") utilizando el decorador @app.get("/"). Esto significa que cuando se realice una solicitud HTTP GET a la ruta principal, se ejecutará la función todos().


@app.get("/")  # ⇠ es un decorador de python
async def todos():  # El async define la funcion root como una funcion asincrona
    return dataset.to_json()  # ⇠ retorna un diccionario de python


@app.get("/pais")  # ⇠ es un decorador de python
async def pais():  # El async define la funcion root como una funcion asincrona
    resultados = set(dataset["Country"])  # ⇠ retorna un diccionario de python
    resultado = [{"id": i, "pais": pais} for i, pais in enumerate(resultados)]
    return resultado  # ⇠ retorna un json de python


@app.get("/iso")  # ⇠ es un decorador de python
async def iso():
    # El async define la funcion root como una funcion asincrona
    iso_records = set(dataset["ISO"])
    return {"iso": iso_records}


@app.get("/region")
async def region():
    # Obtener la lista de regiones únicas del dataset
    region_records = list(set(dataset["Region"]))
    region_records_with_id = [{"id": i, "region": region}
                              for i, region in enumerate(region_records)]
    return region_records_with_id


@app.get("/disasterType")  # ⇠ es un decorador de python
async def disaster_type():
    # El async define la funcion root como una funcion asincrona
    disasterType_records = set(dataset["Disaster Type"])
    return {"disaster_type": disasterType_records}


@app.get("/pais/{pais_id}")
async def leer_pais(pais_id: int):
    resultados = set(dataset["Country"])
    resultados = [{"id": i, "pais": pais} for i, pais in enumerate(resultados)]
    pais = resultados[pais_id]
    return dataset[dataset["Country"] == pais["pais"]].to_json()


@app.get("/prediccion/{pais_id}/{catastrofe}")
async def leer_prediccion(pais_id: int, catastrofe: str):
    resultados = set(dataset["Country"])  # ⇠ retorna un diccionario de python
    resultados = [{"id": i, "pais": pais} for i, pais in enumerate(resultados)]
    pais = resultados[pais_id]
    print(pais)
    prediccion = prediction(pais["pais"], catastrofe.capitalize())
    return pd.DataFrame({"magnitud": prediccion["magnitud"], "probabilidad": prediccion["probabilidad"]}, index = range(0,pais_id)).to_json()

@app.get("/prediccion/{pais_id}")
async def leer_prediccio(pais_id: int, catastrofe: str):
    resultados = set(dataset["Country"])  # ⇠ retorna un diccionario de python
    resultados = [{"id": i, "pais": pais} for i, pais in enumerate(resultados)]
    pais = resultados[pais_id]
    print(pais)
    promedio = []
    catastrofes = ["Drought", "Earthquake", "Epidemic", "Extreme temperature", "Extreme weather", "Flood", "Impact", "Landslide", "Mass movement (dry)", "Volcanic activity", "Wildfire"]
    for i in catastrofes:
        prediccion = prediction(pais["pais"], i)
        promedio.append(prediccion["probabilidad"])
    promedio = np.array(promedio)
    promedio = np.mean(promedio, axis=0)
    print(promedio)
    
    return pd.DataFrame(dict(probabilidad=promedio), index = range(0,pais_id)).to_string()

# Para ejecutar el servidor, en la terminal ejecutar: uvicorn main:app --reload
