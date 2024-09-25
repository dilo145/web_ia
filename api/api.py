from enum import Enum
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

from loguru import logger
import uvicorn
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class PredictionData(BaseModel):
    surface: float

class CityEnum(str, Enum):
        Paris = "Paris"
        Lyon = "Lyon"
        Marseille = "Marseille"

class PredictionDataAppartement(BaseModel):
    surface: float
    nbRooms: int
    nbWindows: int
    price: float
    # year: Optional[int] = None
    # balcony: Optional[bool] = None
    # garage: Optional[bool] = None
    # note: Optional[float] = None
    # city: Optional[CityEnum] = None


model = LinearRegression()

modelSecond = LogisticRegression(max_iter=200)

modelThird = KNeighborsClassifier(n_neighbors=5)

label_encoder = LabelEncoder()

is_model_trained = False

# Endpoint pour obtenir les appartements
@app.get("/suites")
async def get_appartements():
    df = pd.read_csv('../data/suites_cities.csv')

    appartements = df[['surface', 'nbRooms', 'nbWindows', 'price']].to_dict(orient='records')

    return appartements

# Endpoint pour créer un appartement
@app.post("/suites")
async def create_appartement(data: PredictionDataAppartement):
    df = pd.read_csv('../data/suites_cities.csv')

    if not df.empty:
        last_id = df['id'].max()
        new_id = int(last_id) + 1
    else:
        new_id = 1

    new_row = pd.DataFrame([{
        'id': new_id,
        'surface': data.surface,
        'nbRooms': data.nbRooms,
        'nbWindows': data.nbWindows,
        'price': data.price,
        # 'year': data.year if data.year is not None else False,
        # 'balcony': data.balcony if data.balcony is not None else False,
        # 'garage': data.garage if data.garage is not None else False,
        # 'note': data.note if data.note is not None else False,
        # 'city': data.city if data.city is not None else False
    }])

    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv('../data/suites_cities.csv', index=False)

    return {"message": "Appartement ajouté avec succès.", "id": int(new_id)}


@app.post("/predict")
async def predict(surface: float, price: float, city: str):

    df = pd.read_csv('../data/suites_cities_train.csv', sep=',')
    
    df_encoded = pd.get_dummies(df, columns=['city'], prefix='city')

    
    X_note = df_encoded[['city_Lyon', 'city_Marseille', 'city_Paris', 'surface', 'price']]
    y_note = df['note']
    
    X_note_train, X_note_test, y_note_train, y_note_test = train_test_split(X_note, y_note, test_size=0.2, random_state=42)

    model_note = LinearRegression()
    model_note.fit(X_note_train, y_note_train)
    
    y_note_pred = model_note.predict([[surface, price, 1 if city == 'Lyon' else 0, 1 if city == 'Marseille' else 0, 1 if city == 'Paris' else 0]])

    mse_note = mean_squared_error(y_note_test, y_note_pred)
    return "OK"
    rmse_note = np.sqrt(mse_note)
    
    
    X_year = df_encoded[['city_Lyon', 'city_Marseille', 'city_Paris']]
    y_year = df['year']
    
    X_year_train, X_year_test, y_year_train, y_year_test = train_test_split(X_year, y_year, test_size=0.2, random_state=42)
    
    model_year = LinearRegression()
    model_year.fit(X_year_train, y_year_train)
    
    y_year_pred = model_year.predict([[1 if city == 'Lyon' else 0, 1 if city == 'Marseille' else 0, 1 if city == 'Paris' else 0]])
    
    r2_year = model_year.score(X_year_test, y_year_test)
    mse_year = mean_squared_error(y_year_test, y_year_pred)
    rmse_year = np.sqrt(mse_year)
    

    X_garage = df_encoded[['city_Lyon', 'city_Marseille', 'city_Paris', 'price']]
    y_garage = df['garage']
    
    X_garage_train, X_garage_test, y_garage_train, y_garage_test = train_test_split(X_garage, y_garage, test_size=0.2, random_state=42)
    
    model_garage = LogisticRegression()
    model_garage.fit(X_garage_train, y_garage_train)
    
    y_garage_pred = model_garage.predict([[1 if city == 'Lyon' else 0, 1 if city == 'Marseille' else 0, 1 if city == 'Paris' else 0, price]])
    
    conf_matrix = confusion_matrix(y_garage_test, y_garage_pred)
    class_report = classification_report(y_garage_test, y_garage_pred)
    
    return {
        "note_prediction": float(y_note_pred[0]),
        "note_rmse": float(rmse_note),
        "year_prediction": float(y_year_pred[0]),
        "year_r2": float(r2_year),
        "year_rmse": float(rmse_year),
        "garage_prediction": bool(y_garage_pred[0]),
        "garage_report": class_report
    }



















# # Endpoint pour entraîner le modèle

# @app.post("/train")
# async def train():
#     global is_model_trained

#     # Lire le fichier CSV
#     df = pd.read_csv('../data/suites_cities.csv')

#     # Extraction des variables indépendantes et dépendantes
#     X = df[['surface']]  # Variable explicative (surface)
#     y = df['price']  # Variable cible (prix)

#     # Entraînement du modèle
#     model.fit(X, y)
    
#     bins = [0, 150000, 250000, 400000, float('inf')]  # Example thresholds
#     labels = ['low', 'normal', 'high', 'scam']  # Classes
#     df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)


#     X = df[['nbRooms', 'surface', 'nbWindows', 'price']]  # Features
#     y = df['price_category']  # Target variable

#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)
    


#     # Train the model
#     modelSecond.fit(X_train, y_train)

#     # Marquer le modèle comme entraîné
#     is_model_trained = True

#     # Logging avec Loguru
#     logger.info("Modèle entraîné avec succès.")
#     logger.info(f"Coefficients: {model.coef_}, Intercept: {model.intercept_}")

#     def classify_apartment_by_surface(surface):
#         if surface < 40:
#             return 'F1'
#         elif 40 <= surface < 60:
#             return 'F2'
#         elif 60 <= surface < 80:
#             return 'F3'
#         else:
#             return 'F4'

#     df['apartment_type'] = df['surface'].apply(classify_apartment_by_surface)

#     # Encodage des catégories F1, F2, F3, F4
#     df['apartment_type_encoded'] = label_encoder.fit_transform(
#         df['apartment_type'])

#     # Extraction des variables indépendantes (surface et prix) et dépendante (type d'appartement)
#     X = df[['surface',]]  # Variables explicatives
#     y = df['apartment_type_encoded']  # Variable cible (F1, F2, F3, F4)

#     # Diviser les données en ensembles d'entraînement et de test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)
    
#     modelThird.fit(X_train, y_train)


#     return {"message": "Modèle entraîné avec succès."}

# # Endpoint pour prédire un prix en fonction des données d'entrée


# @app.post("/predict")
# async def predict(data: PredictionData):
#     global is_model_trained

#     # Vérifier si le modèle a été entraîné
#     if not is_model_trained:
#         raise HTTPException(
#             status_code=400, detail="Le modèle n'est pas encore entraîné. Veuillez entraîner le modèle d'abord.")

#     X_new = np.array([[data.surface]])

#     # Prédire le prix
#     predicted_price = model.predict(X_new)[0]

#     # Logging avec Loguru
#     logger.info(f"Prédiction faite pour surface: {data.surface}")
#     logger.info(f"Prix prédit: {predicted_price}")

#     return {"predicted_price": predicted_price}


# @app.post("/predict-category")
# async def predictcategory(data: PredictionDataAppartement):
#     X_new = np.array([[data.surface, data.nbRooms, data.nbWindows, data.price],])

#     # Prédire le prix
#     predicted_price = modelSecond.predict(X_new)[0]

#     # Logging avec Loguru
#     logger.info(f"Prédiction faite pour surface: {data.surface}")
#     logger.info(f"Prix prédit: {predicted_price}")

#     return {"predicted_price": predicted_price}


# @app.post("/predict-type")
# async def predictype(data: PredictionData):
#     X_new = np.array(
#         [[data.surface],])

  
#     predicted_type_encoded = modelThird.predict(X_new)[0]
#     predicted_type = label_encoder.inverse_transform(
#         [predicted_type_encoded])[0]

#     return {"predicted_type": predicted_type}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
