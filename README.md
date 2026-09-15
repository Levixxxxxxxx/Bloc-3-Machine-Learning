
# Prédiction du Churn Client — Projet #3

Projet réalisé dans le cadre du cursus **Directeur de projet en intelligence artificielle**
(Année 1, classe 2-AIA01) — L'École Multimédia.

## Contexte

Développement d'un modèle de Machine Learning permettant de prédire le risque de résiliation
(churn) des clients d'une entreprise de télécommunications, à partir du dataset public
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle).

## Structure du projet
├── data/

│ ├── raw/ # Dataset brut (non versionné)

│ └── processed/ # train.csv / test.csv après nettoyage et split

├── notebooks/

│ ├── 01_data_analysis_cleaning.ipynb # Analyse qualité, EDA, nettoyage, split train/test

│ ├── 02_modeling.ipynb # Pipeline, modèles, comparaison, optimisation

│ └── 03_validation.ipynb # Validation finale sur le jeu de test

├── models/

│ ├── random_forest_final.pkl # Modèle final entraîné (pipeline complet)

│ └── random_forest_final_meta.json # Hyperparamètres et score de validation croisée

├── reports/

│ └── figures/ # Graphiques exportés

├── app.py # Dashboard interactif (Streamlit)

├── requirements.txt

└── README.md

## Installation

```bash
python3 -m venv venv
source venv/bin/activate          # venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Reproduire le projet

Exécuter les notebooks dans l'ordre :

1. `01_data_analysis_cleaning.ipynb` — charge le dataset brut, analyse la qualité des données,
   effectue l'EDA, nettoie et sépare train/test (sauvegardés dans `data/processed/`)
2. `02_modeling.ipynb` — construit le pipeline de préprocessing, entraîne et compare plusieurs
   modèles (régression logistique, arbre de décision, Random Forest), optimise les hyperparamètres
   par recherche en grille, sauvegarde le modèle final dans `models/`
3. `03_validation.ipynb` — charge le modèle final et évalue sa performance sur le jeu de test
   (jamais utilisé auparavant), analyse la capacité de généralisation

## Dashboard interactif

```bash
streamlit run app.py
```

Permet d'explorer le taux de churn par variable, de tester l'effet du seuil de décision en
temps réel, de visualiser l'importance des variables, et de simuler la prédiction pour un
client donné.

## Résultats

| Modèle | AUC (CV) | F1 | Précision | Rappel |
|---|---|---|---|---|
| Régression logistique (L2) | 0.846 ± 0.013 | 0.593 | 0.653 | 0.543 |
| Arbre de décision (max_depth=5) | 0.829 ± 0.011 | 0.577 | 0.618 | 0.542 |
| **Random Forest (optimisé)** | **0.848 ± 0.011** | **0.633** | 0.534 | **0.777** |

**Modèle retenu : Random Forest** (`class_weight="balanced"`, `min_samples_leaf=10`,
`n_estimators=200`). AUC de 0.845 sur le jeu de test (contre 0.848 en validation croisée),
confirmant une bonne capacité de généralisation. Le choix de `class_weight="balanced"`
privilégie le rappel (0.79) sur la précision, cohérent avec le coût métier asymétrique du
churn (un client perdu coûte davantage qu'une relance inutile).

## Méthodologie

- Préprocessing (normalisation, encodage) encapsulé dans un `Pipeline` scikit-learn pour
  éviter toute fuite de données pendant la validation croisée
- Validation croisée stratifiée à 5 plis pour toutes les comparaisons de modèles
- Jeu de test isolé dès le départ, utilisé une seule fois pour la validation finale
- Optimisation des hyperparamètres par `GridSearchCV` (critère : AUC-ROC)

## Auteur

Levi Gnakale — classe 2-AIA01, L'École Multimédia