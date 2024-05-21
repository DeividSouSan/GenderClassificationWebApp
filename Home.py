import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.title("Gender Classification Model")

st.header("O que é esse projeto?")

st.markdown("""
Esse projeto é apenas um estudo sobre Machine Learning junto ao desenvolvimento de um web app com o Streamlit. 
            """)

st.header("Objetivo")

st.markdown("""
O objetivo desse projeto foi colocar em prática o que eu conhecia sobre Machine Learning e criar uma aplicação que tornasse esse modelo útil.

O modelo consiste no algoritmo K-Nearest Neighbor aplicado a um conjunto de dados onde o nome e o sexo de uma pessoa são as principais colunas. A ideia é conseguir classificar o sexo a partir do nome da pessoa.
        """)

st.header("Como usar?")

st.markdown("""
A aplicação possui duas páginas principais (sem contar essa, a Home). A primeira chamada 'Train and Test' onde o usuário insere um conjunto de dados (qualquer um, porém tem um no github pronto) com o nome e sexo da pessoa e configura (de maneira limitada) o algoritmo para gerar seu próprio modelo (é completamente normal se o treinamento demorar). Nessa mesma página ainda é possível baixar o modelo após ele ser treinado.

A outra página é o 'Use Model' onde o modelo baixado pelo usuário poderá ser utilizado para fazer previsões reais. A depender da acurácio do modelo gerado as previsões podem ser boas ou não.
        """)