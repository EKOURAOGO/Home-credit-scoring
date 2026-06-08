# Modélisation du risque de crédit — Home Credit Default Risk

> Spécialité : Big Data et Data Science 

**Auteurs :** Emmanuel KOURAOGO
**Directeur :** Baptiste de La ROBERTIE, Docteur en informatique

---

## Sujet

**Optimisation de l'évaluation du risque de crédit : Comparaison entre la régression logistique et le Machine Learning pour la prédiction du défaut.**

---

## Résultats clés

| Modèle | AUC-ROC | Rappel (défaut) | Gini |
|--------|---------|-----------------|------|
| CatBoost | **0.769** | **67.9%** | 0.538 |
| Logit sklearn OHE | 0.759 | 68.8% | 0.518 |
| XGBoost | 0.751 | 43.6% | 0.502 |
| Random Forest | 0.754 | 0.3% | 0.508 |
| Gradient Boosting | 0.749 | 67.8% | 0.498 |
| Logit statsmodels | 0.749 | **2%** | 0.498 |
| LightGBM | 0.713 | 0% | 0.426 |

**Conclusion principale :** CatBoost surpasse tous les modèles (AUC = 0.769). La régression logistique classique n'identifie que 2% des défauts — totalement inadaptée à la prédiction opérationnelle.

---

## Dataset — Home Credit Default Risk (Kaggle)

- **307 511 observations** d'entraînement
- **7 tables relationnelles** : `application_train`, `bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`
- Variable cible binaire `TARGET` : défaut (1) / non-défaut (0)
- Déséquilibre marqué : **8% de défauts** — pondération `scale_pos_weight ≈ 11.39`

---

## Méthodologie

### Feature engineering (128 → 189 variables)
- Agrégation multi-tables : historique bureau de crédit, paiements passés, applications précédentes
- Variables comportementales : `avg_payment_delay`, `prev_remaining_debt_ratio`, `bureau_active_count`
- Ratios financiers : `revenu/annuité`, `bureau_max_credit_ratio`
- Imputation : médiane pour les numériques, catégorie "Unknown" pour les catégorielles

### Prétraitement
- Encodage One-Hot Encoding (OHE) des variables catégorielles
- Normalisation (moyenne=0, écart-type=1)
- Pondération des classes pour corriger le déséquilibre (scale_pos_weight)

### Modèles comparés
- Régression logistique (statsmodels — inférence) vs sklearn (prédiction)
- Random Forest (Breiman, 2001)
- Gradient Boosting, XGBoost, LightGBM, CatBoost
- Optimisation Grid Search + validation croisée

### Métriques d'évaluation
- AUC-ROC (principale), Gini = 2×AUC−1
- Matrice de confusion, Rappel, Précision, F1-score
- Courbe de Lift, Gains cumulés, KS Statistic

---

## Hypothèses validées

**H1** — Les modèles ML surpassent la régression logistique en performance prédictive : **validée** (CatBoost AUC=0.769 vs Logit classique AUC=0.749, rappel 68% vs 2%)

**H2** — Le prétraitement est le principal moteur de performance : **validée** (la régression logistique sklearn optimisée atteint AUC=0.759, proche de CatBoost)

---

## Structure du repo

```
CODE-MEMOIRE/
├── Code_Memoire.ipynb    # Notebook complet (feature engineering + benchmark 7 modeles)
└── README.md
```

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib seaborn missingno jupyter
jupyter notebook Code_Memoire.ipynb
```

---

## Stack technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-red?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-green?style=flat-square)
![CatBoost-blue](https://img.shields.io/badge/CatBoost-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

---

## Auteurs

| Nom | Profil |
|-----|--------|
| Emmanuel KOURAOGO | [GitHub](https://github.com/EKOURAOGO) |

ESG Finance — Titre RNCP Niveau 7, Expert en Finance de Marché, 2024-2025
