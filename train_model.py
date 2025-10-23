import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# --- 1. CONFIGURATION ---
DATA_PATH = os.path.join("data", "credit_risk_dataset.csv")
MODEL_OUTPUT_PATH = os.path.join("models", "credit_scoring_pipeline.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- 2. FONCTIONS DE TRAITEMENT ET D'ENTRAÎNEMENT ---

def load_and_clean_data(file_path: str) -> pd.DataFrame | None:
    """
    Charge les données depuis un fichier CSV, les nettoie et les prépare.
    
    Args:
        file_path (str): Chemin vers le fichier de données.
    
    Returns:
        pd.DataFrame | None: DataFrame nettoyé ou None si le fichier n'est pas trouvé.
    """
    print(f"Chargement des données depuis '{file_path}'...")
    try:
        credit = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERREUR: Le fichier de données '{file_path}' est introuvable.")
        print("Veuillez vous assurer que le fichier 'credit_risk_dataset.csv' se trouve bien dans le dossier 'data/'.")
        return None
    
    # Suppression des doublons
    credit = credit.drop_duplicates()
    
    # Imputation des valeurs manquantes pour l'ancienneté
    median_emp_length = credit['person_emp_length'].median()
    credit['person_emp_length'] = credit['person_emp_length'].fillna(median_emp_length)
    
    # Suppression des variables causant une fuite de données (data leakage)
    credit = credit.drop(columns=["loan_grade", "loan_int_rate"])
    
    # Suppression des outliers/erreurs de saisie
    credit = credit[credit['person_age'] <= 100]
    credit = credit[credit['person_emp_length'] <= 100]
    
    # Suppression de la variable fortement corrélée pour éviter la multicolinéarité
    credit = credit.drop(columns=["cb_person_cred_hist_length"])
    
    print("Nettoyage et prétraitement des données terminés.")
    return credit

def build_and_train_pipeline(df: pd.DataFrame) -> Pipeline:
    """
    Construit et entraîne un pipeline complet (prétraitement + modèle XGBoost).
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données nettoyées.
    
    Returns:
        Pipeline: Le pipeline entraîné et prêt à être sauvegardé.
    """
    print("Construction et entraînement du pipeline...")
    
    # Séparation des features (X) et de la cible (y)
    X = df.drop(columns='loan_status')
    y = df['loan_status']
    
    # Division en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Définition des listes de variables numériques et catégorielles
    var_nominal = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
    var_num = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_percent_income']
    
    # Pipeline pour les variables numériques
    numeric_pipeline = Pipeline(steps=[
        ('log_transform', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())
    ])
    
    # Préprocesseur qui applique les transformations aux bonnes colonnes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, var_num),
            ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), var_nominal)
        ],
        remainder='passthrough'
    )
    
    # Meilleurs hyperparamètres pour XGBoost trouvés via l'optimisation bayésienne
    best_params_xgb = {
        'colsample_bytree': 0.5123060710796715, 
        'gamma': 0.16314440758761042, 
        'learning_rate': 0.034290853230315534, 
        'max_depth': 16, 
        'min_child_weight': 1.0, 
        'reg_alpha': 5.0, 
        'reg_lambda': 0.61793339983887,
        'n_estimators' : 1000
 
    }
    
    # Création du pipeline final
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            random_state=RANDOM_STATE, 
            eval_metric='logloss', 
            **best_params_xgb
        ))
    ])
    
    # Entraînement du pipeline sur les données d'entraînement
    final_pipeline.fit(X_train, y_train)
    
    print("Entraînement du pipeline terminé.")
    return final_pipeline


# --- 3. POINT D'ENTRÉE DU SCRIPT ---
if __name__ == "__main__":
    print("--- Début du script d'entraînement du modèle de scoring ---")
    
    # Étape 1: Charger et nettoyer les données
    df_credit = load_and_clean_data(DATA_PATH)
    
    if df_credit is not None:
        # Étape 2: Entraîner le pipeline
        trained_pipeline = build_and_train_pipeline(df_credit)
        
        # Étape 3: Sauvegarder le pipeline
        os.makedirs("models", exist_ok=True)
        joblib.dump(trained_pipeline, MODEL_OUTPUT_PATH)
        
        print("-" * 50)
        print(f"Pipeline d'entraînement sauvegardé avec succès dans : {MODEL_OUTPUT_PATH}")
        print("--- Script terminé ---")