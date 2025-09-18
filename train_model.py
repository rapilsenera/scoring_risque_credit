import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

DATA_PATH = os.path.join("data", "credit_risk_dataset.csv")
MODEL_OUTPUT_PATH = os.path.join("models", "xgb_model.joblib")
PREPROCESSOR_OUTPUT_PATH = os.path.join("models", "preprocessor.joblib")

# --- 1. CHARGEMENT ET NETTOYAGE DES DONNÉES ---
def load_and_clean_data(file_path):
    """Charge et nettoie les données depuis un fichier CSV."""
    try:
        credit = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erreur: Le fichier de données '{file_path}' n'a pas été trouvé.")
        print("Veuillez vous assurer que le fichier 'credit_risk_dataset.csv' est bien dans le dossier 'data/'.")
        return None

    credit = credit.drop_duplicates()
    
    # Imputation des valeurs manquantes
    median_emp_length = credit['person_emp_length'].median()
    credit['person_emp_length'] = credit['person_emp_length'].fillna(median_emp_length)
    
    # Suppression des variables avec fuite de données et outliers
    credit = credit.drop(columns=["loan_grade", "loan_int_rate"])
    credit = credit[credit['person_age'] <= 100]
    credit = credit[credit['person_emp_length'] <= 100]
    
    # Suppression de la colonne corrélée (si elle existe encore après nettoyage)
    if "cb_person_cred_hist_length" in credit.columns:
        credit = credit.drop(columns=["cb_person_cred_hist_length"])
    
    return credit

# --- 2. ENTRAÎNEMENT DU MODÈLE ---
def train_model(df):
    """Entraîne le modèle et retourne le modèle et le préprocesseur."""
    # Séparation X et y
    X = df.drop(columns='loan_status')
    y = df['loan_status']
    
    # Division Entraînement / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    # Définition du préprocesseur
    var_nominal = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
    var_num = ['person_age','person_income','person_emp_length','loan_amnt','loan_percent_income']
    
    numeric_pipeline = Pipeline(steps=[
        ('log_transform', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, var_num),
            ('nom', OneHotEncoder(drop='first', sparse_output=False), var_nominal)
        ],
        remainder='passthrough'
    )
    
    # Appliquer le préprocesseur et SMOTE
    X_train_processed = preprocessor.fit_transform(X_train)
    smote = SMOTE(sampling_strategy='minority', random_state=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

    # Entraînement du meilleur modèle (XGBoost) avec les meilleurs paramètres
    best_params_xgb = {
        'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.1, 
        'max_depth': 9, 'n_estimators': 250, 'reg_lambda': 1, 'subsample': 0.8
    }
    
    model = xgb.XGBClassifier(random_state=3, eval_metric='logloss', **best_params_xgb)
    model.fit(X_train_smote, y_train_smote)
    
    return model, preprocessor

# --- POINT D'ENTRÉE DU SCRIPT ---
if __name__ == "__main__":
    print("Début de l'entraînement...")
    
    df_credit = load_and_clean_data(DATA_PATH)
    
    if df_credit is not None:
        # Entraîner le modèle et récupérer le préprocesseur
        final_model, final_preprocessor = train_model(df_credit)
        
        os.makedirs("models", exist_ok=True)
        
        joblib.dump(final_model, MODEL_OUTPUT_PATH)
        joblib.dump(final_preprocessor, PREPROCESSOR_OUTPUT_PATH)
        
        print(f"Modèle sauvegardé dans : {MODEL_OUTPUT_PATH}")
        print(f"Préprocesseur sauvegardé dans : {PREPROCESSOR_OUTPUT_PATH}")
        print("Entraînement terminé.")