# 📋 TODO LIST - PREPROCESSING VARIABLES ORDINALES ET CATÉGORIELLES

## 🎯 Objectif
Créer une classe `OrdinalPreprocessor`et `CategoricalPreprocessor` avec des méthodes indépendantes pour préprocesser les 96 variables ordinales et 70 variables catégorielles respectivement, en vue de prédire **MathScore** (variable cible à NE PAS modifier).

---

## 📊 CONTEXTE DU DATASET

### Variables à traiter :
- **96 variables ordinales** (31.2% du dataset)
  - 22 sur le contexte socio-économique familial
  - 13 sur l'environnement de classe
  - 11 sur l'utilisation des TIC
  - 50 autres réparties sur 9 domaines

- **70 variables catégorielles** (22.7% du dataset)
  - 28 variables générales (métadonnées)
  - 42 variables thématiques réparties sur 10 domaines

### Variables encodées avec haute cardinalité :
- **STRATUM** : 1316 strates (urbain/rural, public/privé, régions)
- **OCOD** : 620 codes de professions
- **CNT/CNTRYID** : 80 pays (information redondante)

---

## ✅ TODO LIST DÉTAILLÉE

### 🔵 PHASE 1 : ANALYSE EXPLORATOIRE (EDA)

#### ☐ 1.1 - `load_reference_dictionaries(glossaire_path: str) -> dict`
**Objectif** : Charger les 5 feuilles de référence du glossaire
- Charger CNT, CNTRYID, STRATUM, ISCEDP, OCOD
- Créer des dictionnaires de mapping {code: label}
- Stocker dans `self.reference_dicts`
- **Output** : Dict avec clés ['CNT', 'CNTRYID', 'STRATUM', 'ISCEDP', 'OCOD']

#### ☐ 1.2 - `analyze_ordinal_distributions(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Analyser les distributions des 96 variables ordinales
- Pour chaque variable ordinale :
  - Compter les modalités uniques
  - Calculer le % de valeurs manquantes
  - Identifier les modalités les plus fréquentes
  - Détecter les valeurs aberrantes (hors plage attendue)
- **Output** : DataFrame avec colonnes [variable, n_unique, missing_%, top_3_values, potential_issues]

#### ☐ 1.3 - `analyze_categorical_distributions(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Analyser les distributions des 70 variables catégorielles
- Pour chaque variable catégorielle :
  - Compter la cardinalité (nombre de catégories)
  - Calculer le % de valeurs manquantes
  - Identifier les catégories rares (< 1% ou seuil personnalisé)
  - Mesurer le déséquilibre des classes
- **Output** : DataFrame avec colonnes [variable, cardinality, missing_%, rare_categories, imbalance_ratio]

#### ☐ 1.4 - `identify_redundant_variables(df: pd.DataFrame, threshold: float = 0.95) -> list`
**Objectif** : Détecter les variables redondantes
- Calculer corrélations de Spearman pour ordinales
- Calculer Cramér's V pour catégorielles
- Identifier CNT vs CNTRYID (100% redondants)
- **Output** : Liste de tuples [(var1, var2, correlation_score)]

#### ☐ 1.5 - `detect_ordinal_scales(df: pd.DataFrame) -> dict`
**Objectif** : Identifier les types d'échelles ordinales
- Détecter échelles de Likert (3, 4, 5, 7 points)
- Détecter échelles de fréquence (Never, Rarely, Sometimes, Often, Always)
- Détecter échelles de quantité (0, 1-2, 3-5, 6-10, 11+)
- **Output** : Dict {variable: scale_type, ex: 'likert_5', 'frequency_6', 'quantity_ranges'}

---

### 🟡 PHASE 2 : GESTION DES VALEURS MANQUANTES

#### ☐ 2.1 - `flag_missing_values(df: pd.DataFrame, missing_indicators: list) -> pd.DataFrame`
**Objectif** : Identifier et harmoniser les codes de valeurs manquantes
- Codes courants PISA : -99, -98, -97, 97, 98, 99, "N/A", "Missing", ""
- Remplacer tous par np.nan
- Créer variables indicatrices si > 10% missing : `var_name_is_missing`
- **Output** : DataFrame avec valeurs manquantes harmonisées + variables indicatrices

#### ☐ 2.2 - `impute_ordinal_missing_median(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Imputer variables ordinales par la médiane
- Pour variables ordinales avec < 20% missing
- Imputation par médiane (préserve le caractère ordinal)
- Option : imputation stratifiée par pays (CNT) si pertinent
- **Output** : DataFrame avec ordinales imputées

#### ☐ 2.3 - `impute_ordinal_missing_mode(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Imputer variables ordinales par le mode
- Pour variables ordinales très déséquilibrées
- Imputation par mode (valeur la plus fréquente)
- **Output** : DataFrame avec ordinales imputées

#### ☐ 2.4 - `impute_categorical_missing_mode(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Imputer variables catégorielles par le mode
- Pour variables catégorielles avec < 20% missing
- Imputation par mode global ou stratifié
- **Output** : DataFrame avec catégorielles imputées

#### ☐ 2.5 - `create_missing_category(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Créer une catégorie "Missing" pour variables catégorielles
- Pour variables catégorielles avec > 20% missing
- Ajouter une modalité explicite "Unknown" ou "Missing"
- **Output** : DataFrame avec nouvelle catégorie

#### ☐ 2.6 - `drop_high_missing_variables(df: pd.DataFrame, threshold: float = 0.5) -> tuple`
**Objectif** : Supprimer variables avec trop de valeurs manquantes
- Identifier variables avec > 50% missing (ou seuil personnalisé)
- Les exclure du dataset
- **Output** : (DataFrame nettoyé, liste des variables supprimées)

---

### 🟢 PHASE 3 : TRAITEMENT DES CATÉGORIES RARES

#### ☐ 3.1 - `group_rare_categories(df: pd.DataFrame, var: str, threshold: float = 0.01) -> pd.DataFrame`
**Objectif** : Regrouper catégories rares en "Other"
- Pour une variable catégorielle
- Regrouper modalités représentant < 1% (ou seuil) en "Other"
- Conserver mapping pour interprétabilité
- **Output** : DataFrame avec catégories regroupées

#### ☐ 3.2 - `reduce_stratum_dimensionality(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Réduire les 1316 strates en features exploitables
- Parser STRATUM pour extraire :
  - `stratum_location` : Urban / Rural
  - `stratum_region` : North / Center / South / etc.
  - `stratum_type` : Public / Private
  - `stratum_country` : Code pays (3 lettres)
- Supprimer STRATUM original
- **Output** : DataFrame avec 4 nouvelles variables + suppression STRATUM

#### ☐ 3.3 - `group_occupations_by_major_group(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Regrouper les 620 professions en grands groupes ISCO
- Utiliser le 1er chiffre du code OCOD pour créer 10 groupes :
  - 0: Armed forces
  - 1: Managers
  - 2: Professionals
  - 3: Technicians
  - 4: Clerical support
  - 5: Service and sales
  - 6: Skilled agricultural
  - 7: Craft workers
  - 8: Plant operators
  - 9: Elementary occupations
- **Output** : DataFrame avec OCOD remplacé par OCOD_major_group

#### ☐ 3.4 - `resolve_cnt_cntryid_redundancy(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Supprimer la redondance entre CNT et CNTRYID
- Vérifier corrélation parfaite
- Garder CNT (plus lisible : codes 3 lettres)
- Supprimer CNTRYID
- **Output** : DataFrame sans CNTRYID

---

### 🟠 PHASE 4 : ENCODAGE DES VARIABLES

#### ☐ 4.1 - `encode_ordinal_variables(df: pd.DataFrame, mapping_dict: dict = None) -> pd.DataFrame`
**Objectif** : Encoder les variables ordinales en préservant l'ordre
- Utiliser OrdinalEncoder de sklearn
- Créer mappings explicites pour échelles Likert, fréquences
- Exemple : {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
- Stocker encoders dans `self.encoders`
- **Output** : DataFrame avec ordinales encodées en int

#### ☐ 4.2 - `encode_binary_categorical(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Encoder variables catégorielles binaires
- Pour variables avec exactement 2 modalités (ex: Gender, OECD Yes/No)
- Encoder en 0/1 avec LabelEncoder
- **Output** : DataFrame avec binaires encodées

#### ☐ 4.3 - `onehot_encode_low_cardinality(df: pd.DataFrame, max_categories: int = 10) -> pd.DataFrame`
**Objectif** : One-Hot Encoding pour variables à faible cardinalité
- Pour variables catégorielles avec ≤ 10 modalités
- Utiliser pd.get_dummies ou OneHotEncoder
- Nommer colonnes : `var_name_category`
- **Output** : DataFrame avec colonnes one-hot créées

#### ☐ 4.4 - `target_encode_high_cardinality(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Target Encoding pour variables à haute cardinalité
- Pour variables avec > 10 modalités (CNT, langues, etc.)
- Encoder par moyenne de MathScore pour chaque catégorie
- Ajouter régularisation (smoothing) pour catégories rares
- Attention au data leakage : utiliser cross-validation
- **Output** : DataFrame avec target encoding appliqué

#### ☐ 4.5 - `frequency_encode_categorical(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Frequency Encoding (alternative au Target Encoding)
- Encoder par fréquence d'apparition de chaque catégorie
- Moins risqué que target encoding (pas de leakage)
- **Output** : DataFrame avec frequency encoding appliqué

---

### 🔴 PHASE 5 : VALIDATION ET CONTRÔLE QUALITÉ

#### ☐ 5.1 - `validate_no_missing_after_preprocessing(df: pd.DataFrame) -> bool`
**Objectif** : Vérifier qu'il n'y a plus de valeurs manquantes
- Compter les NaN restants
- Lever une exception si NaN détectés
- **Output** : True si OK, raise ValueError sinon

#### ☐ 5.2 - `validate_dtypes_after_encoding(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Vérifier les types de données après encodage
- Ordinales encodées → int ou float
- Catégorielles encodées → int ou float
- Pas de type 'object' sauf si voulu
- **Output** : DataFrame de validation avec [column, expected_dtype, actual_dtype, status]

#### ☐ 5.3 - `check_target_variable_unchanged(df_before: pd.DataFrame, df_after: pd.DataFrame) -> bool`
**Objectif** : Vérifier que MathScore n'a pas été modifié
- Comparer MathScore avant et après preprocessing
- Lever exception si différences détectées
- **Output** : True si identique, raise ValueError sinon

#### ☐ 5.4 - `generate_preprocessing_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict`
**Objectif** : Générer un rapport de preprocessing
- Nombre de variables avant/après
- Variables supprimées et raison
- Variables créées (one-hot, indicatrices missing)
- Statistiques d'encodage
- **Output** : Dict avec toutes les métadonnées

#### ☐ 5.5 - `detect_data_leakage_risk(df: pd.DataFrame) -> list`
**Objectif** : Détecter les risques de data leakage
- Identifier si target encoding fait sans CV
- Identifier si imputation utilise statistiques globales
- Identifier si normalisation faite sur tout le dataset
- **Output** : Liste des warnings de leakage potentiel

---

### 🟣 PHASE 6 : PIPELINE ET SAUVEGARDE

#### ☐ 6.1 - `create_preprocessing_pipeline(steps: list) -> Pipeline`
**Objectif** : Créer un pipeline sklearn réutilisable
- Enchaîner les transformations dans l'ordre
- Utiliser ColumnTransformer pour appliquer transformations par type
- **Output** : Pipeline sklearn fitted

#### ☐ 6.2 - `save_encoders_and_mappings(filepath: str) -> None`
**Objectif** : Sauvegarder les encoders pour réutilisation
- Pickler les OrdinalEncoder, LabelEncoder, OneHotEncoder
- Sauvegarder les mappings de référence
- Sauvegarder les listes de variables par type
- **Output** : Fichier .pkl

#### ☐ 6.3 - `export_preprocessed_data(df: pd.DataFrame, filepath: str) -> None`
**Objectif** : Exporter le dataset préprocessé
- Sauvegarder en CSV ou Parquet
- Inclure métadonnées dans un fichier séparé
- **Output** : Fichiers data + metadata

#### ☐ 6.4 - `transform_new_data(df_new: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Appliquer le preprocessing à de nouvelles données
- Charger les encoders sauvegardés
- Appliquer les mêmes transformations
- Gérer les nouvelles catégories inconnues
- **Output** : DataFrame transformé

---

## 🎯 ORDRE DE PRIORITÉ D'EXÉCUTION

### Sprint 1 - Analyse (Semaine 1)
1. 1.1 → 1.2 → 1.3 → 1.4 → 1.5

### Sprint 2 - Nettoyage (Semaine 1-2)
2. 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6

### Sprint 3 - Réduction (Semaine 2)
3. 3.1 → 3.2 → 3.3 → 3.4

### Sprint 4 - Encodage (Semaine 2-3)
4. 4.1 → 4.2 → 4.3 → (4.4 OU 4.5)

### Sprint 5 - Validation (Semaine 3)
5. 5.1 → 5.2 → 5.3 → 5.4 → 5.5

### Sprint 6 - Production (Semaine 3)
6. 6.1 → 6.2 → 6.3 → 6.4

---

## ⚠️ POINTS D'ATTENTION CRITIQUES

### 🚨 Data Leakage
- **Target encoding** : OBLIGATOIRE d'utiliser cross-validation
- **Imputation** : Calculer statistiques UNIQUEMENT sur train set
- **Scaling** : Fit sur train, transform sur test

### 🚨 Gestion de MathScore (Cible)
- **NE JAMAIS** modifier, imputer, ou encoder MathScore
- Vérifier après chaque transformation avec `check_target_variable_unchanged()`
- Exclure MathScore de toutes les transformations

### 🚨 Haute Cardinalité
- STRATUM (1316) et OCOD (620) : **RÉDUCTION OBLIGATOIRE**
- Ne JAMAIS faire de one-hot sur ces variables
- Privilégier feature engineering intelligent

### 🚨 Variables Redondantes
- CNT vs CNTRYID : supprimer l'un des deux
- Vérifier corrélations avant encodage

### 🚨 Préservation de l'Information Ordinale
- Ne JAMAIS one-hot des variables ordinales
- Utiliser OrdinalEncoder avec mapping explicite
- Documenter l'ordre des modalités

---

## 📚 LIBRAIRIES NÉCESSAIRES

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    OrdinalEncoder, LabelEncoder, OneHotEncoder, StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import pickle
import json
```

---

## 📊 LIVRABLES ATTENDUS

1. **Classe Python** : `OrdinalCategoricalPreprocessor` avec toutes les méthodes
2. **Notebook d'exemples** : Démonstration de chaque méthode
3. **Dataset preprocessé** : Fichier final prêt pour modélisation
4. **Documentation** : Rapport de preprocessing détaillé
5. **Encoders sauvegardés** : Fichiers .pkl pour réutilisation
6. **Tests unitaires** : Validation de chaque méthode

---

## 🎓 BONNES PRATIQUES À RESPECTER

✅ **Programmation Orientée Objet** : Créer des classes si c'est pertinent.
✅ **Noms de fonctions explicites** : `encode_ordinal_variables` pas `encode_vars`
✅ **Docstrings complètes** : Paramètres, returns, exemples
✅ **Type hints** : `def func(df: pd.DataFrame) -> pd.DataFrame`
✅ **Logging** : Logger chaque transformation importante
✅ **Traçabilité** : Conserver metadata de chaque transformation
✅ **Tests** : Valider sur échantillon avant full dataset
✅ **Modularité** : Chaque fonction fait UNE chose
✅ **Réutilisabilité** : Code applicable à de nouvelles données

---

## 📝 NOTES FINALES

- Cette TODO list est **exhaustive mais flexible** : adapter selon les données réelles
- Certaines méthodes peuvent être optionnelles selon les analyses de Phase 1
- Prioriser la **qualité** sur la vitesse : un bon preprocessing = 80% du succès du modèle
- **Documenter** toutes les décisions prises et les justifier

**Prêt à commencer le développement ! 🚀**