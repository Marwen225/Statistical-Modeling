import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Titre de l'application
st.title("Modélisation Statistique des Caractéristiques Musicales")

# Explication de l'application
st.write("""
Bienvenue dans notre analyse statistique des caractéristiques musicales. Nous allons explorer les corrélations entre les différentes caractéristiques musicales 
et la popularité des chansons, et appliquer une modélisation statistique pour prédire la popularité.
""")

# Téléchargement du fichier CSV
fichier = st.file_uploader("Téléchargez le fichier de données musicales (.csv)", type=["csv"])

if fichier is not None:
    # Chargement du fichier
    music_data = pd.read_csv(fichier)

    # Nettoyage des données
    st.write("### Aperçu des données")
    st.write(music_data.head())

    # Informations sur les données
    st.write("### Informations sur les colonnes")
    st.write(music_data.info())

    # Description statistique
    st.write("### Description statistique des données")
    st.write(music_data.describe())

    # Suppression des colonnes inutiles
    music_data_cleaned = music_data.drop(columns=['Unnamed: 0'])

    # Gestion des valeurs manquantes
    st.write("### Valeurs manquantes par colonne")
    st.write(music_data_cleaned.isnull().sum())

    columns_with_missing_values = ['Track Name', 'Artists', 'Album Name']
    music_data_cleaned[columns_with_missing_values] = music_data_cleaned[columns_with_missing_values].fillna('Inconnu')

    # Distribution des scores de popularité
    st.write("### Distribution des scores de popularité")
    plt.figure(figsize=(10, 6))
    sns.histplot(music_data_cleaned['Popularity'], bins=20, kde=True)
    plt.title('Distribution des scores de popularité')
    st.pyplot(plt)

    # Matrice de corrélation
    st.write("### Matrice de corrélation")
    plt.figure(figsize=(12, 10))
    correlation_matrix = music_data_cleaned.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de corrélation des caractéristiques numériques')
    st.pyplot(plt)

    # Analyse des relations entre les caractéristiques et la popularité
    st.write("### Analyse des caractéristiques musicales par rapport à la popularité")

    # Sélection des caractéristiques pertinentes
    features = ['Danceability', 'Energy', 'Loudness', 'Acousticness', 'Valence']
    plt.figure(figsize=(15, 10))

    for i, feature in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(x=music_data_cleaned[feature], y=music_data_cleaned['Popularity'])
        plt.title(f'{feature} vs. Popularité')
        plt.xlabel(feature)
        plt.ylabel('Popularité')

    st.pyplot(plt)

    # Préparation des données pour la modélisation
    st.write("### Préparation des données pour la modélisation statistique")

    # Conversion de la colonne 'Explicit' en format numérique
    music_data_cleaned['Explicit'] = music_data_cleaned['Explicit'].astype(int)

    # Sélection des caractéristiques pour le modèle
    features = ['Danceability', 'Energy', 'Loudness', 'Acousticness', 'Valence', 'Explicit', 'Key', 'Mode', 'Speechiness', 'Instrumentalness', 'Tempo']
    X = music_data_cleaned[features]
    y = music_data_cleaned['Popularity']

    # Standardisation des caractéristiques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Division du dataset en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialisation et entraînement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluation du modèle
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"### Erreur quadratique moyenne (MSE) : {mse:.2f}")
    st.write(f"### Coefficient de détermination (R²) : {r2:.2f}")

    # Affichage des coefficients du modèle
    coefficients = pd.Series(model.coef_, index=features)
    st.write("### Coefficients du modèle de régression linéaire")
    st.write(coefficients)

    # Visualisation des résultats
    st.write("### Visualisation des prédictions vs. réalité")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Valeurs réelles de popularité")
    plt.ylabel("Valeurs prédites de popularité")
    plt.title("Comparaison des valeurs réelles et prédites")
    st.pyplot(plt)
