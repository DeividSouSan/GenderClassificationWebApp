import pickle

import pandas as pd
import streamlit as st
from faker import Faker
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def load_data(csv_file: str) -> pd.DataFrame:
    """
    Loads a CSV and turns it into a pd.Dataframe.

    Args:
        csv_file: a valid CSV (Comma Separated Values) file or path that contains the columns 'Name' and 'Gender'.

    Returns:
        None

    Raises:
        KeyError: if the CSV file doesn't contains the needed data.
    """

    df = pd.read_csv(csv_file)

    try:
        df = df.loc[:, ["Name", "Gender"]]
    except KeyError as err:
        raise KeyError("Columns 'Name' or 'Gender' are not in the dataframe.")

    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Process the data inside the dataframe to generate the data that the model will use to train and to test.

    Args:
        df: a valid (with the needed columns 'Name' and 'Gender') pd.Dataframe.

    Returns:
        X: pd.Dataframe that contains all the names and their features (the count of chars used).
        y: pd.Series that contains the name and their sex represented by 0 (Fem) and 1 (Masc).
    """

    # Seleciona os dados necessários do DataFrame
    df = df.loc[:, ["Name", "Gender"]]

    # Inicio da Verificação de Oversampling
    masc_names = df[df["Gender"] == "M"].shape[0]
    fem_names = df[df["Gender"] == "F"].shape[0]

    needed_number_of_names = abs(masc_names - fem_names)

    if needed_number_of_names > 0:
        faker = Faker()
        generated_names = []

        if masc_names < fem_names:
            name = faker.first_name_male()
            sex = "M"

        elif fem_names < masc_names:
            name = faker.first_name_fem()
            sex = "F"

        for _ in range(needed_number_of_names):
            generated_names.append([name, sex])

        generated_names_df = pd.DataFrame(generated_names, columns=["Name", "Gender"])

        df = pd.concat([df, generated_names_df], ignore_index=True)

    # One-hot Encoding para altear 'F' e 'M' para 0 e 1
    df["Gender"] = df["Gender"].replace({"F": 0, "M": 1})

    # Convertendo os nomes em 'tokens' (números)
    names = df["Name"].values  # Esse vetor é considerado um doc de texto

    vectorizer = CountVectorizer(analyzer="char")
    vectorizer = vectorizer.fit(names)

    # Salvando o Vetorizador preenchido para uso posterior
    with open("vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    # Criando a matriz final com os dados de cada nome
    count_matrix = vectorizer.transform(names).toarray()

    feature_chars = vectorizer.get_feature_names_out()

    count_letter_df = pd.DataFrame(
        data=count_matrix, columns=feature_chars, index=names
    )

    # Features
    X = count_letter_df

    # Label
    y = df["Gender"]

    return X, y


def train_model(X, y, test_size: float, random_state: int, n_neighbors: int):
    """
    Splits the X and y generated from preprocess_data() into Train and Test values to train the model and evaluate it's accuracy.

    Args:
        X: pd.Dataframe with all names and their features.
        y: pd.Series with the respective X outputs.
        test_size: percentage of test samples.
        random_state: random seed.
        n_neighbors: KNN (K-Nearest Neighbors) configuration.

    Return:
        And dict with the keys 'Model' containing the model itself and 'Accuracy' containing the model accuracy.

    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Treinando o Modelo
    classifier.fit(X_train.values, y_train.values)

    # Testando o Modelo
    score = classifier.score(X_test, y_test)

    return {"Model": classifier, "Accuracy": score}


def preprocesses_input(name: str):
    """
    Handles user's input processing it to become a valida dataframe (with a row with the name and columns with the chars/features).

    Args:
        name: string with the user's input.

    Returns:
        A pd.Dataframe containing the vectorized input (the name and its features).
    """
    # Recupera o vetorizador já preenchido
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    # Gera a matriz como nome e suas features
    matrix = vectorizer.transform([name]).toarray()

    # Recupera todas as features do vetorizador
    chars = vectorizer.get_feature_names_out()

    # Gera o dataframe como nome e suas features
    df_sample = pd.DataFrame(matrix, columns=chars, index=[name])

    return df_sample
