# Projet de scoring de crédit

Analyse du risque de défaut et création d'un modèle prédictif à partir du jeu de données [credit_risk_dataset sur Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).

## Contexte

Dans le secteur financier, évaluer avec précision le risque de défaut est primordial pour la rentabilité et la stabilité des établissements de crédit. Ce projet vise à développer un modèle de scoring de crédit performant en utilisant des techniques de Machine Learning pour prédire si un client est susceptible de faire défaut sur son prêt.

## Structure du dépôt

- **/app**: Contient le code de l'application web (`app.py`) et le fichier de style (`style.css`).
- **/data**: Contient le jeu de données `credit_risk_dataset.csv`.
- **/models**: Stocke le préprocesseur (`processor.joblib`) et le modèle XGBoost entraîné (`xgb_model.joblib`).
- **/notebooks**: Renferme le notebook Jupyter `analyse_scoring_credit.ipynb` qui détaille l'analyse exploratoire et la modélisation.
- **train_model.py**: Script Python permettant de ré-entraîner le modèle et de sauvegarder les artefacts.
- **requirements.txt**: Liste des dépendances Python nécessaires au projet.

## Installation

Pour exécuter ce projet en local, suivez ces étapes :

1.  **Clonez le dépôt :**
    ```bash
    git clone https://github.com/rapilsenera/scoring_risque_credit.git
    cd scoring_risque_credit
    ```

2.  **Créez un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1.  **Analyse et modélisation** :
    Pour explorer l'analyse complète, ouvrez et exécutez le notebook `notebooks/analyse_scoring_credit.ipynb`.

2.  **(Optionnel) Ré-entraîner le modèle** :
    Exécutez le script suivant à la racine du projet pour entraîner le modèle XGBoost et sauvegarder le préprocesseur et le modèle dans le dossier `/models`.
    ```bash
    python train_model.py
    ```

3.  **Lancer l'application** :
    Pour lancer l'interface utilisateur, exécutez le script `app.py`.
    ```bash
    python app/app.py
    ```
    Ouvrez ensuite votre navigateur à l'adresse indiquée.

## Modélisation

Trois modèles de classification ont été comparés : Régression Logistique, Forêt Aléatoire et XGBoost. Le modèle XGBoost a été sélectionné pour ses performances supérieures, avec un **AUC ROC de 0.896** et une **accuracy de 89.2%** sur l'ensemble de test.

## Technologies utilisées

- Python 3.12.7
- Pandas & NumPy pour la manipulation des données
- Matplotlib & Seaborn pour la visualisation
- Scikit-learn pour le prétraitement et la modélisation
- XGBoost pour le modèle final
- Imblearn pour le rééchantillonnage (SMOTE)