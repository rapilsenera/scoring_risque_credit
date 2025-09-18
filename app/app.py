import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CSS_PATH = os.path.join(BASE_DIR, "style.css") 
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgb_model.joblib")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")

# --- Configuration de la page ---
st.set_page_config(
    page_title="Scoring de crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJECTION CSS ---
def load_css(file_name):
    """Charge un fichier CSS local pour styliser l'application."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Le fichier CSS '{file_name}' n'a pas été trouvé. L'application utilisera les styles par défaut.")

# Correction du chemin pour le CSS
load_css(CSS_PATH)


# --- CHARGEMENT DU MODÈLE ET DU PRÉPROCESSEUR ---
@st.cache_resource
def load_artefacts():
    """Charge le modèle XGBoost et le préprocesseur depuis les fichiers."""
    try:
        # Correction des chemins pour charger les modèles
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"Erreur de chargement des modèles : {e}")
        st.error(f"Vérifiez que les fichiers existent aux chemins suivants depuis la racine du projet :")
        st.error(f"- Modèle : {os.path.relpath(MODEL_PATH, PROJECT_ROOT)}")
        st.error(f"- Préprocesseur : {os.path.relpath(PREPROCESSOR_PATH, PROJECT_ROOT)}")
        return None, None

model, preprocessor = load_artefacts()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.title("À propos du projet")
    st.info(
        "Cette application utilise un modèle **XGBoost** pour prédire la probabilité "
        "de défaut de paiement d'un client et aider à la décision d'octroi de crédit."
    )

    st.markdown("---")
    st.header("Créé par")
    st.markdown("**Antsa Ramanantsalama**") 
    st.markdown(
        "[Mon profil LinkedIn](https://www.linkedin.com/in/antsa-ramanantsalama-9788192a0/)"
    )
    # Pensez à mettre le vrai lien vers votre dépôt GitHub ici
    st.markdown(
        "[Voir le projet sur GitHub](https://github.com/VOTRE_NOM_UTILISATEUR/credit-scoring-project)"  
    )
    st.markdown("---")

# --- PAGE PRINCIPALE ---
if model is not None:
    st.title("Outil de scoring de crédit")
    st.markdown(
        "Renseignez les informations du demandeur pour obtenir une évaluation instantanée du risque de défaut."
    )

    # --- FORMULAIRE DE SAISIE ---
    with st.container(border=True):
        st.subheader("Informations sur le demandeur")
        
        col1, col2 = st.columns(2)

        with col1:
            person_age = st.number_input("Âge du demandeur", min_value=18, max_value=100, value=25, step=1)
            person_home_ownership = st.selectbox(
                "Statut immobilier",
                options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
                help="Le type de propriété du logement."
            )
            person_emp_length = st.slider("Ancienneté professionnelle (années)", min_value=0, max_value=50, value=5, step=1)
            
        with col2:
            person_income = st.number_input("Revenu annuel (€)", min_value=0, value=50000, step=1000)
            loan_intent = st.selectbox(
                "Motif du prêt",
                options=['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
            )
            loan_amnt = st.number_input("Montant du prêt demandé (€)", min_value=500, value=10000, step=100)

        st.subheader("Historique de crédit")
        cb_person_default_on_file = st.radio(
            "Le demandeur a-t-il un historique de défaut de paiement ?",
            ('Non', 'Oui'), index=0, horizontal=True
        )
        cb_person_default_on_file_map = 'Y' if cb_person_default_on_file == 'Oui' else 'N'

    st.write("") # 
    
    # --- BOUTON D'ÉVALUATION ---
    if st.button("Évaluer le risque du dossier", type="primary", use_container_width=True):

        # Calcul dynamique du ratio prêt/revenu
        loan_percent_income = (loan_amnt / person_income) if person_income > 0 else 0

        # Préparation des données pour le modèle
        input_data = {
            'person_age': [person_age], 'person_income': [person_income],
            'person_home_ownership': [person_home_ownership], 'person_emp_length': [person_emp_length],
            'loan_intent': [loan_intent], 'loan_amnt': [loan_amnt],
            'loan_percent_income': [loan_percent_income], 'cb_person_default_on_file': [cb_person_default_on_file_map]
        }
        input_df = pd.DataFrame(input_data)

        # Simulation du calcul
        with st.spinner('Analyse du dossier par l\'IA...'):
            time.sleep(1.5)

            # Prétraitement et prédiction
            input_processed = preprocessor.transform(input_df)
            prediction = model.predict(input_processed)
            prediction_proba = model.predict_proba(input_processed)
            probability_default = prediction_proba[0][1]

        # --- AFFICHAGE DU RÉSULTAT ---
        st.markdown("---")
        st.header("Résultat de l'analyse")
        
        if prediction[0] == 0:
            decision_text = "✅ Prêt approuvé"
            decision_help = "Le risque de défaut est considéré comme faible."
            badge_color = "green"
        else:
            decision_text = "❌ Prêt rejeté"
            decision_help = "Le risque de défaut est considéré comme élevé."
            badge_color = "red"

        with st.container(border=True):
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric(
                    label="Probabilité de défaut",
                    value=f"{probability_default:.2%}"
                )
                
            with col_res2:
                st.subheader(decision_text, help=decision_help)
                if prediction[0] == 0:
                    st.progress(float(probability_default), text=f"Niveau de risque : {probability_default:.0%}")
                else:
                    st.progress(1.0, text=f"Risque élevé : {probability_default:.0%}")


        with st.expander("Afficher les détails techniques de la prédiction"):
            st.write(f"**Ratio prêt/revenu (calculé) :** {loan_percent_income:.2f}")
            st.write(f"**Probabilité de non-défaut (Classe 0) :** {prediction_proba[0][0]:.2%}")
            st.write(f"**Probabilité de défaut (Classe 1) :** {prediction_proba[0][1]:.2%}")
            st.write("**Données saisies pour le modèle :**")
            st.dataframe(input_df)
else:
    st.title("Initialisation de l'application...")
    st.warning("Le modèle de scoring n'est pas chargé. Veuillez vérifier la configuration.")