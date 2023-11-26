from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузим данные из CSV-файла
data = pd.read_csv('I:\Институт\Статистическое моделирование\Computers.csv')
# Удалим столбец "Unnamed: 0"
data = data.drop("Unnamed: 0", axis=1)
# Применим One-Hot Encoding к признаку "cd"
data = pd.get_dummies(data, columns=['cd'], drop_first=True)
# Применим One-Hot Encoding к столбцам multi и premium
data = pd.get_dummies(data, columns=['multi', 'premium'])

# Разделим данные на признаки (X) и целевую переменную (y)
X = data.drop('price', axis=1)
y = data['price']

# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создадим модель градиентного бустинга и обучим её
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Эндпоинт для предсказания цены
@app.post("/predict/")
def predict_price(speed: int = Form(...), hd: int = Form(...), ram: int = Form(...), screen: int = Form(...), ads: int = Form(...), trend: int = Form(...), cd: str = Form(...), multi: str = Form(...), premium: str = Form(...)):
    # Преобразование текстовых значений в столбце cd с использованием One-Hot Encoding
    input_data = pd.get_dummies(pd.DataFrame({
        'speed': [speed],
        'hd': [hd],
        'ram': [ram],
        'screen': [screen],
        'ads': [ads],
        'trend': [trend],
        'cd': [cd],
        'multi': [multi],
        'premium': [premium],
    }), columns=['cd', 'multi', 'premium'], drop_first=True)

    # Предсказание цены
    predicted_price = model.predict(input_data)[0]

    # Возвращаем результат
    result = {"predicted_price": predicted_price}
    return JSONResponse(content=result)
