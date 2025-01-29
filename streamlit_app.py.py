import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import plotly.express as px
import plotly.graph_objects as go
from keras.models import load_model

# Configuración de la app
st.title("Predicción de precios con LSTM")
st.caption("Por: Fernando Guzman")
st.write("Esta aplicación utiliza un modelo de redes neuronales LSTM para predecir los precios de cierre de acciones basándose en datos históricos. Long Short-Term Memory (LSTM) es un tipo de red neuronal recurrente diseñada para aprender patrones de largo plazo en datos secuenciales. Su arquitectura permite recordar información durante largos periodos, lo que la hace ideal para la predicción de series temporales, como los precios de acciones. El precio de prediccion de cierre es el de un dia despues de tu fecha final.")




st.sidebar.header("Configuración de la Predicción")
ticker = st.sidebar.text_input("Ingrese el ticker de la acción", value="^GSPC")
start_date = st.sidebar.date_input("Fecha de inicio", value=pd.to_datetime("2000-01-01"), min_value=pd.to_datetime("2000-01-01"), max_value=pd.to_datetime("2025-12-31"))
end_date = st.sidebar.date_input("Fecha final", value=pd.to_datetime("2025-01-28"), min_value=pd.to_datetime("2000-01-01"), max_value=pd.to_datetime("2025-12-31"))



if st.sidebar.button("Ejecutar Modelo"):
   
    df = yf.download(ticker, start=start_date, end=end_date)
    st.write("### Datos Históricos")
    st.write(df.tail())

   
    st.write("### Gráfico del Precio de Cierre")
    fig = px.line(df, x=df.index, y="Close", title="Closing Price")
    st.plotly_chart(fig)
    
    
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * 0.8)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:training_data_len , :]
    x_train, y_train = [], []
    
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    
    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    st.write(f"### RMSE: {rmse}")
    st.write("El error cuadrático medio (RMSE) mide la cantidad de error que hay entre dos conjuntos de datos.")
    
 
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    st.write("### Gráfico de Predicción")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
    fig2.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Validation'))
    fig2.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))
    fig2.update_layout(title='Model Predictions', xaxis_title='Date', yaxis_title='Close Price USD')
    st.plotly_chart(fig2)
    
    
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    st.write(f"### Predicción del próximo precio de cierre: {pred_price[0][0]}")
    
    
    st.write("### Footnote")
    st.write("El proceso sigue los siguientes pasos:")
    st.write("1. Se descargan los datos históricos desde Yahoo Finance.")
    st.write("2. Se visualiza el precio de cierre a lo largo del tiempo.")
    st.write("3. Se procesan los datos para preparar el modelo.")
    st.write("4. Se construye un modelo LSTM para procesar la serie temporal.")
    st.write("5. Se entrena el modelo con los datos disponibles.")
    st.write("6. Se generan predicciones y se evalúa su precisión con RMSE.")
    st.write("7. Se visualizan los datos de entrenamiento, validación y predicción.")
