import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import StringIO
import datetime
from  matplotlib import pyplot as plt

# Pour le modèle LSTM (reste basique pour l'exemple)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Configuration de page Streamlit
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Titre principal
st.title("📈 Stock Price Prediction avec LSTM")

st.markdown("""
Cette application prédit les prix futurs des actions en utilisant un modèle **LSTM** 
(Long Short-Term Memory). Entrez simplement le symbole boursier et laissez l'IA faire le reste !
""")

# ==================== BARRE LATÉRALE ====================
st.sidebar.header("⚙️ Paramètres")

# Saisie du symbole boursier
ticker = st.sidebar.text_input(
    "Entrez le symbole boursier",
    value="AAPL",
    help="Ex: AAPL (Apple), GOOGL (Google), MSFT (Microsoft)"
)

# Slider pour le nombre de jours à prédire
days = st.sidebar.slider(
    "Nombre de jours à prédire",
    min_value=7,
    max_value=60,
    value=30,
    step=1,
    help="Combien de jours dans le futur voulez-vous prédire ?"
)

# Checkbox pour afficher les données historiques
show_data = st.sidebar.checkbox("Afficher les données historiques", value=True)

# ==================== BOUTON DE LANCEMENT ====================
st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Lancer la prédiction", use_container_width=True)


def fetch_data_csv_yahoo(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:#ticker : Symbole boursier (ex: AAPL, GOOGL)
    """
    Télécharge l'historique des prix depuis l'endpoint CSV de Yahoo Finance.
    Retourne un tuple (DataFrame, error_msg). Si tout va bien, error_msg == ''.
    La fonction tente d'abord une requête sur la page d'historique pour
    récupérer les cookies nécessaires puis télécharge le CSV via le endpoint.
    """
    error_msg = ""
    try:
        period1 = int(pd.Timestamp(start).timestamp())
        period2 = int(pd.Timestamp(end).timestamp())
    except Exception as e:
        return pd.DataFrame(), f"Invalid date: {e}"

    download_url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedCl¨£ose=true"
    )

    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                      " Chrome/117.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Hit the history page first to establish cookies
    try:
        history_url = f"https://finance.yahoo.com/quote/{ticker}/history?p={ticker}"
        session.get(history_url, headers=headers, timeout=10)
    except Exception:
        # not fatal, continue to try download
        pass
 
    try:
        resp = session.get(download_url, headers=headers, timeout=10)
    except Exception as e:
        return pd.DataFrame(), f"Request error: {e}"

    # If Yahoo blocks (401 / unauthorized), essayer une source alternative (Stooq)
    if resp.status_code == 401 or 'unauthorized' in (resp.text or '').lower():
        # Stooq expects lowercase ticker with .us for US stocks
        tik = ticker.lower()
        if not tik.endswith('.us'):
            tik_stooq = f"{tik}.us"
        else:
            tik_stooq = tik
        stooq_url = f"https://stooq.com/q/d/l/?s={tik_stooq}&i=d"
        try:
            resp2 = requests.get(stooq_url, timeout=10)
        except Exception as e:
            return pd.DataFrame(), f"Yahoo 401 and Stooq request error: {e}"

        if resp2.status_code != 200 or not resp2.text.strip():
            return pd.DataFrame(), f"Yahoo 401 and Stooq HTTP {resp2.status_code}"

        try:
            df = pd.read_csv(StringIO(resp2.text), parse_dates=['Date'], index_col='Date')
            return df, ''
        except Exception as e:
            return pd.DataFrame(), f"Stooq CSV parse error: {e}"

    if resp.status_code != 200 or resp.text.strip() == "":
        snippet = resp.text[:500].replace('\n', ' ') if resp.text else ''
        return pd.DataFrame(), f"HTTP {resp.status_code} - Response snippet: {snippet}"

    try:
        df = pd.read_csv(StringIO(resp.text), parse_dates=['Date'], index_col='Date')
    except Exception as e:
        return pd.DataFrame(), f"CSV parse error: {e}"

    return df, ''

# ==================== EXÉCUTION PRINCIPALE ====================
if run_button:

    with st.spinner(f"⏳ Téléchargement des données pour {ticker}..."):
        # Step 1: Télécharger les données via l'endpoint CSV Yahoo
        data, err = fetch_data_csv_yahoo(
            ticker,
            start='2020-01-01',
            end=datetime.datetime.today().strftime('%Y-%m-%d')
        )

        if err:
            # Afficher un message d'erreur diagnostic pour aider à comprendre pourquoi
            st.error(f"❌ Impossible de récupérer les données : {err}")

        if not data.empty and 'Close' in data.columns:
            data = data[['Close']]

    if data.empty:
        if not err:
            st.error(f"❌ Erreur : Le symbole '{ticker}' n'existe pas. Vérifiez l'orthographe !")
        # on arrête ici si pas de données
    else:
        # Afficher les données historiques si demandé
        if show_data:
            st.subheader("📊 Données historiques")
            st.write(f"Total des jours de données: {len(data)}")
            
            # Onglets pour choisir entre graphique et tableau
            tab1, tab2 = st.tabs(["📈 Graphique", "📋 Tableau"])
            
            with tab1:
                fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
                ax_hist.plot(data.index, data['Close'], color='blue', linewidth=2)
                ax_hist.set_xlabel('Date')
                ax_hist.set_ylabel('Prix ($)')
                ax_hist.set_title(f'Historique des prix - {ticker}')
                ax_hist.grid(True, alpha=0.3)
                fig_hist.autofmt_xdate(rotation=45)
                st.pyplot(fig_hist)
            
            with tab2:
                st.dataframe(data.tail(20), width='stretch')

        with st.spinner("🔄 Entraînement du modèle..."):
            # Step 2: Normaliser les données
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Step 3: Préparer les données d'entraînement
            training_data_len = int(np.ceil(len(scaled_data) * 0.8))
            train_data = scaled_data[:training_data_len, :]

            x_train, y_train = [], []

            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Step 4: Créer le modèle LSTM
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(60, 1)),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mean_squared_error')

            # Step 5: Entraîner le modèle
            model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

        with st.spinner("🔮 Génération des prédictions..."):
            # Step 6: Prédire les jours futurs
            last_60_days = scaled_data[-60:]
            x_future = last_60_days.reshape((1, 60, 1))

            future_predictions = []

            for _ in range(days):
                pred = model.predict(x_future, verbose=0)
                future_predictions.append(pred[0, 0])
                x_future = np.append(x_future[:, 1:, :], [[pred[0]]], axis=1)

            # Dénormaliser les prédictions
            future_predictions = scaler.inverse_transform(
                np.array(future_predictions).reshape(-1, 1)
            )

            # Step 7: Créer le DataFrame des prédictions
            forecast_dates = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=1),
                periods=days,
                freq='B'
            )

            forecast = pd.DataFrame(
                future_predictions,
                index=forecast_dates,
                columns=['Prediction']
            )

        # ==================== RÉSULTATS ====================
        st.success("✅ Prédiction terminée avec succès !")

        # Onglets pour les résultats
        result_tab1, result_tab2 = st.tabs(["📈 Graphique", "📋 Tableau"])
        
        with result_tab1:
            st.subheader("📈 Graphique des prédictions")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['Close'], label='Prix historique', color='blue', linewidth=2)
            ax.plot(forecast.index, forecast['Prediction'], label='Prédictions', color='red', linewidth=2, linestyle='--')
            ax.set_xlabel('Date')
            ax.set_ylabel('Prix ($)')
            ax.set_title(f'Prédiction des prix pour {ticker}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate(rotation=45)
            st.pyplot(fig)
        
        with result_tab2:
            st.subheader("🎯 Détail des prédictions")
            # Convertir l'index en colonne et réinitialiser l'index
            forecast_display = forecast.reset_index()
            forecast_display.columns = ['Date', 'Prix Prédit']
            forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(forecast_display, width='stretch', hide_index=True)
        # Statistiques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prix actuel", f"${data['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("Prédiction J+1", f"${forecast['Prediction'].iloc[0]:.2f}")
        with col3:
            st.metric("Prix max prédit", f"${forecast['Prediction'].max():.2f}")
        with col4:
            st.metric("Prix min prédit", f"${forecast['Prediction'].min():.2f}")



# ==================== SECTION D'INFO ====================
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ À propos")
st.sidebar.info("""
**LSTM Prediction** utilise une intelligence artificielle pour prédire 
les prix des actions. Le modèle apprend des tendances passées 
pour estimer les prix futurs.

⚠️ *Avertissement: Ces prédictions ne garantissent pas les résultats futurs.*
""")
