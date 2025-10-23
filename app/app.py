import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

# --- 1. CONFIGURATION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CSS_PATH = os.path.join(BASE_DIR, "style.css") 
# Mise à jour pour charger le pipeline unique
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "credit_scoring_pipeline.joblib")

# --- 2. CONFIGURATION DE LA PAGE STREAMLIT ---
st.set_page_config(
    page_title="Scoring de crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. INJECTION DU CSS PERSONNALISÉ ---
def load_css(file_name):
    """Charge un fichier CSS local pour styliser l'application."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Le fichier CSS '{file_name}' n'a pas été trouvé. L'application utilisera les styles par défaut.")

load_css(CSS_PATH)

# --- 4. CHARGEMENT DU PIPELINE ---
@st.cache_resource
def load_pipeline():
    """Charge le pipeline de scoring de crédit (préprocesseur + modèle)."""
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        return pipeline
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier du pipeline ('{os.path.basename(PIPELINE_PATH)}') est introuvable.")
        st.error(f"Veuillez exécuter le script 'train_model.py' pour le générer.")
        st.error(f"Chemin attendu : {PIPELINE_PATH}")
        return None

pipeline = load_pipeline()

# --- 5. BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.title("À propos du projet")
    st.info(
        "Cette application utilise un modèle **XGBoost** pour prédire la probabilité "
        "de défaut de paiement et aider à la décision d'octroi de crédit."
    )
    st.markdown("---")
    st.header("Créé par")
    st.markdown("**Antsa Ramanantsalama**") 
    st.markdown("[Mon profil LinkedIn](https://www.linkedin.com/in/antsa-ramanantsalama-9788192a0/)")
    st.markdown("[Voir le projet sur GitHub](https://github.com/rapilsenera/scoring_risque_credit)")
    st.markdown("---")

# --- 6. PAGE PRINCIPALE ---
if pipeline is not None:
    st.title("Outil de scoring de crédit")
    st.markdown("Renseignez les informations du demandeur pour obtenir une évaluation instantanée du risque de défaut.")

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
            person_income = st.number_input("Revenu annuel (€)", min_value=1, value=50000, step=1000)
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

    st.write("") # Espace
    
    # --- BOUTON D'ÉVALUATION ---
    if st.button("Évaluer le risque du dossier", type="primary", use_container_width=True):

        # Calcul dynamique du ratio prêt/revenu
        loan_percent_income = loan_amnt / person_income

        # Préparation des données dans un DataFrame
        input_data = {
            'person_age': [person_age], 
            'person_income': [person_income],
            'person_home_ownership': [person_home_ownership], 
            'person_emp_length': [person_emp_length],
            'loan_intent': [loan_intent], 
            'loan_amnt': [loan_amnt],
            'loan_percent_income': [loan_percent_income], 
            'cb_person_default_on_file': [cb_person_default_on_file_map]
        }
        input_df = pd.DataFrame(input_data)

        # Simulation du calcul
        with st.spinner('Analyse du dossier par l\'IA...'):
            time.sleep(1.5)

            # --- Prédiction simplifiée avec le pipeline ---
            # L'appel au pipeline se fait en une seule étape !
            # Il gère le prétraitement ET la prédiction.
            prediction = pipeline.predict(input_df)
            prediction_proba = pipeline.predict_proba(input_df)
            probability_default = prediction_proba[0][1]

        # --- AFFICHAGE DU RÉSULTAT ---
        st.markdown("---")
        st.header("Résultat de l'analyse")
        
        is_rejected = prediction[0] == 1

        with st.container(border=True):
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.metric(
                    label="Probabilité de défaut",
                    value=f"{probability_default:.2%}"
                )
                
            with col_res2:
                if is_rejected:
                    st.subheader("❌ Prêt rejeté", help="Le risque de défaut est considéré comme élevé.")
                    st.progress(float(probability_default), text=f"Risque élevé : {probability_default:.0%}")
                else:
                    st.subheader("✅ Prêt approuvé", help="Le risque de défaut est considéré comme faible.")
                    st.progress(float(probability_default), text=f"Niveau de risque : {probability_default:.0%}")

        with st.expander("Afficher les détails techniques de la prédiction"):
            st.write(f"**Ratio prêt/revenu (calculé) :** {loan_percent_income:.2f}")
            st.write(f"**Probabilité de non-défaut (Classe 0) :** {prediction_proba[0][0]:.2%}")
            st.write(f"**Probabilité de défaut (Classe 1) :** {prediction_proba[0][1]:.2%}")
            st.write("**Données saisies pour le modèle :**")
            st.dataframe(input_df)

else:
    st.title("Initialisation de l'application...")
    st.warning("Le pipeline de scoring n'a pas pu être chargé. Veuillez vérifier la configuration et les logs.")