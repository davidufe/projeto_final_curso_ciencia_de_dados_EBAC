import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from pycaret.classification import load_model

# Funções de pré-processamento
def missing(df):
    df.fillna(0, inplace=True)
    return df

def substituir_outliers(df, colunas=None, estrategia="mediana"):
    if colunas is None:
        colunas = df.select_dtypes(include=["number"]).columns

    df_tratado = df.copy()

    for coluna in colunas:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        if estrategia == "mediana":
            mediana = df[coluna].median()
            df_tratado[coluna] = np.where(
                (df[coluna] < limite_inferior) | (
                    df[coluna] > limite_superior),
                mediana,
                df[coluna],
            )

        elif estrategia == "limite":
            df_tratado[coluna] = np.where(
                df[coluna] < limite_inferior, limite_inferior,
                np.where(df[coluna] > limite_superior,
                         limite_superior, df[coluna])
            )

    return df_tratado

def aplicar_pca(df, n_componentes=5):
    colunas_numericas = df.select_dtypes(include=["number"])
    pca = PCA(n_components=n_componentes, random_state=42)
    componentes_principais = pca.fit_transform(colunas_numericas)
    df_pca = pd.DataFrame(
        componentes_principais,
        columns=[f"PCA_{i+1}" for i in range(n_componentes)],
        index=df.index
    )
    return df_pca

def dummies(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_2 = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_2

# Definir pipeline
pipeline = Pipeline([
    ('missing', FunctionTransformer(missing, validate=False)),
    ('dummies', FunctionTransformer(dummies, validate=False)),
    ('outliers', FunctionTransformer(lambda df: substituir_outliers(
        df, estrategia="mediana"), validate=False)),
    ('pca', FunctionTransformer(lambda df: aplicar_pca(
        df, n_componentes=5), validate=False))
])

# Configuração do Streamlit
st.set_page_config(
    page_title="Projeto Final - Classificação Binária",
    page_icon="input/telmarketing_icon.png"
)

st.image("input/Bank-Branding.jpg", use_column_width=True)

st.title("Projeto Final - EBAC")
st.write("Este é um projeto de conclusão do curso de Cientista de Dados da EBAC.")
st.write("O objetivo é simular a aplicação de um modelo de Machine Learning para prever potenciais inadimplentes.")

# Upload do arquivo
uploaded_file = st.file_uploader(
    "Faça o upload de um arquivo CSV para análise", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Exibir dados brutos antes do pré-processamento
    st.write("### Dados brutos antes do pré-processamento:")
    st.dataframe(df.head())

    # Aplicar pipeline
    st.write("### Processando os dados...")
    df_processado = pipeline.fit_transform(df)

    # Carregar o modelo treinado
    st.write("### Carregando o modelo...")
    modelo = load_model("input/Final GBC Model")

    # Previsões
    st.write("### Realizando previsões...")
    predicoes = modelo.predict(df_processado)

    # Adicionar coluna de previsões
    df['mau'] = predicoes

    # Exibir resultados
    st.write("### Dados com previsões:")
    st.dataframe(df.head())

    # Botão para baixar resultados
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar resultados em CSV",
        data=csv,
        file_name="resultados_com_previsoes.csv",
        mime="text/csv"
    )

    # Gráfico de proporção
    st.write("### Gráfico de proporção de clientes")
    proporcoes = df['mau'].value_counts(normalize=True) * 100
    st.bar_chart(proporcoes)

