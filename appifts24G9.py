import requests
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


# Cargo el dataset (Modificar teniendo en cuenta el dataset que se usa si se usa algun encoders y si eliminan algunas columnas)
data_path = 'muerte causas_2019-2021.csv'
df_original = pd.read_csv(data_path)

# Aplicar label encoding utilizando pd.factorize()
LE_HHSRegion, _ = pd.factorize(df_original['HHSRegion'])
LE_AgeGroup, _ = pd.factorize(df_original['AgeGroup'])

# Agregar la columna al DataFrame original
df_original['LE_HHSRegion'] = LE_HHSRegion
df_original['LE_AgeGroup'] = LE_AgeGroup

# eliminar columnas innecesarias
df = df_original.drop(['AnalysisDate','Note','flag_allcause','flag_natcause','flag_sept','flag_neopl','flag_diab','flag_alz','flag_inflpn','flag_clrd','flag_otherresp','flag_nephr','flag_otherunk','flag_hd','flag_stroke','flag_cov19mcod','flag_cov19ucod'], axis=1)


# Apply the function
df = df.fillna(0)


st.title("Visualizador Interactivo con Filtros")

# Identificar columnas numéricas y categóricas
col_num = df.select_dtypes(include='number').columns.tolist()
col_cat = df.select_dtypes(exclude='number').columns.tolist()

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
#    El DataFrame está vacío
if df.empty:
    print("El DataFrame está vacío.")
    error=1

#    Las columnas especificadas no existen en el DataFrame.
if col_x not in df.columns or col_y not in df.columns:
    print("Las columnas especificadas no existen en el DataFrame.")
    error=1

#    si tomo columnas categoricas las cambio por columnas donde se aplico label encoding.

if col_x == "AgeGroup" or col_y == "AgeGroup":
    col_x = "LE_AgeGroup"
if col_x == "HHSRegion" or col_y == "HHSRegion":
    col_x = "LE_HHSRegion"


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

        # Mostrar regresion lineal
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_test[col_x], y=y_test, ax=ax, label='Datos Reales')
        sns.lineplot(x=X_test[col_x], y=y_pred, color='red', ax=ax, label='Línea de Regresión')
        ax.set_title('Regresión Lineal')
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig) 

    elif modelo == 'Regresión logística':
        X_train, X_test, y_train, y_test = train_test_split(df[[col_x]], (df[col_y] > 0).astype(int), test_size=0.2, random_state=42)
        if len(set(y_train)) == 1:
            st.error("No hay suficientes datos para entrenar el modelo de regresión logística. Por favor, seleccione otra variable o modelo.")
        else:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(f'Precisión: {model.score(X_test, y_test):.2f}')

            # Mostrar regresion Logistica (probabilidades)
            # necesitamos predecir probabilidades para mostrar
            y_prob = model.predict_proba(X_test)[:, 1] # probabilidad de la clase positiva

            fig, ax = plt.subplots(figsize=(10, 6))
            # Ordenar valores para una curva logística más suave
            sort_idx = X_test[col_x].argsort()
            X_test_sorted = X_test[col_x].iloc[sort_idx]
            y_prob_sorted = y_prob[sort_idx]

            sns.scatterplot(x=X_test[col_x], y=y_test, ax=ax, alpha=0.6, label='Datos Reales (0 o 1)')
            sns.lineplot(x=X_test_sorted, y=y_prob_sorted, color='red', ax=ax, label='Probabilidad Predicha')
            ax.set_title('Regresión Logística')
            ax.set_xlabel(col_x)
            ax.set_ylabel('Probabilidad (Clase 1)')
            ax.set_ylim(-0.1, 1.1)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    elif modelo == 'K-vecinos más cercanos':
        X_train, X_test, y_train, y_test = train_test_split(df[[col_x]], df[col_y], test_size=0.2, random_state=42)
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f'MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}')

        # Mostrar Regression vecinos
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_test[col_x], y=y_test, ax=ax, label='Datos Reales')
        # Ordene X_test para dibujar una línea suave para la predicción de vecinos
        sort_idx = X_test[col_x].argsort()
        sns.lineplot(x=X_test[col_x].iloc[sort_idx], y=y_pred[sort_idx], color='red', ax=ax, label='Predicción KNN')
        ax.set_title('Regresión K-Vecinos Más Cercanos')
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
