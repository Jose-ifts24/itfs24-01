import requests
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Cargo el dataset 
#data_path = '../DS/muerte causas_2019-2021.csv'
data_path = 'muerte causas_2019-2021.csv'
#df = pd.read_csv(data_path, sep=';')
df_original = pd.read_csv(data_path)
df = df_original.drop(['AnalysisDate','Note','flag_allcause','flag_natcause','flag_sept','flag_neopl','flag_diab','flag_alz','flag_inflpn','flag_clrd','flag_otherresp','flag_nephr','flag_otherunk','flag_hd','flag_stroke','flag_cov19mcod','flag_cov19ucod'], axis=1)

# Apply the function
df = df.fillna(0)


st.title("Visualizador Interactivo con Filtros")

# Identificar columnas numéricas y categóricas
col_num = df.select_dtypes(include='number').columns.tolist()
col_cat = df.select_dtypes(exclude='number').columns.tolist()
#col_cat = df.columns[1]

st.sidebar.header("Seleccionar Filtros")

# Aplicar filtros para cada columna categórica
filtros = {}
for col in col_cat:
    valores_unicos = df[col].dropna().unique().tolist()
    seleccionados = st.sidebar.multiselect(f"Filtrar por {col}:", opciones := sorted(valores_unicos), default=opciones)
    filtros[col] = seleccionados

# Filtrar el DataFrame
for col, valores in filtros.items():
    df = df[df[col].isin(valores)]

# Selección de columnas para el gráfico
st.subheader("Configuración del gráfico")

col_x = st.selectbox("Selecciona la columna para el eje X", df.columns)
col_y = st.selectbox("Selecciona la columna para el eje Y (numérica)", col_num)
col_color = st.selectbox("Agrupar por (color)", ["Ninguna"] + col_cat)

# Tipo de gráfico
tipo = st.radio("Tipo de gráfico", ["Dispersión", "Línea", "Barras"])
#tipo = st.radio("Tipo de gráfico", ["Dispersión", "Barras"])

# Crear gráfico
if not df.empty:
    color_arg = None if col_color == "Ninguna" else col_color

    if tipo == "Dispersión":
        fig = px.scatter(df, x=col_x, y=col_y, color=color_arg, title=f"{col_y} vs {col_x}")
    elif tipo == "Línea":
        fig = px.line(df, x=col_x, y=col_y, color=color_arg, title=f"{col_y} vs {col_x}")
    elif tipo == "Barras":
        fig = px.bar(df, x=col_x, y=col_y, color=color_arg, title=f"{col_y} vs {col_x}")

    st.plotly_chart(fig)
else:
    st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados.")

# Selección del tipo de modelo de machine learning
modelo = st.selectbox('Seleccione el modelo de machine learning', ['Regresión lineal', 'Regresión logística', 'K-vecinos más cercanos'])
error=0
if df.empty:
#    raise ValueError("El DataFrame está vacío.")
    print("El DataFrame está vacío.")
    error=1
if col_x not in df.columns or col_y not in df.columns:
#    raise ValueError("Las columnas especificadas no existen en el DataFrame.")
    print("Las columnas especificadas no existen en el DataFrame.")
    error=1

# Entrenar modelo
if error == 0:
    if modelo == 'Regresión lineal':
        X_train, X_test, y_train, y_test = train_test_split(df[[col_x]], df[col_y], test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f'MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}')
    elif modelo == 'Regresión logística':
        X_train, X_test, y_train, y_test = train_test_split(df[[col_x]], (df[col_y] > 0).astype(int), test_size=0.2, random_state=42)
        if len(set(y_train)) == 1:
            st.error("No hay suficientes datos para entrenar el modelo de regresión logística. Por favor, seleccione otra variable o modelo.")
        else:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(f'Precisión: {model.score(X_test, y_test):.2f}')
    elif modelo == 'K-vecinos más cercanos':
        X_train, X_test, y_train, y_test = train_test_split(df[[col_x]], df[col_y], test_size=0.2, random_state=42)
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f'MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}')
