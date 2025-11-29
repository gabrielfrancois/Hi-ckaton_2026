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

### 🔵 PHASE 1 : ANALYSE EXPLORATOIRE (EDA) - Dans Jupyter Notebook
**Objectif** : Explorer les données réelles pour comprendre les distributions et guider le preprocessing

#### [V] 1.1 - Créer notebook `preprocessing/01_eda_ordinal_categorical.ipynb`
- Charger échantillon de `data/X_train.csv` (10 000-50 000 lignes)
- Charger dictionnaires de référence du `data/Glossaire.xlsx`

#### [V] 1.2 - Identifier les variables ordinales et catégorielles dans les données réelles
- Utiliser le classification_variable.xlsx pour classifier les variables
- Créer listes: `ordinal_vars` (96 vars) et `categorical_vars` (70 vars)
- Vérifier cohérence avec la structure réelle du dataset

#### [V] 1.3 - Analyser distributions des variables ordinales
- Cardinalité (nombre de valeurs uniques)
- % de valeurs manquantes par variable
- Détecter types d'échelles (Likert, fréquence, quantité)
- Identifier valeurs aberrantes ou codes spéciaux (-99, 97, 98, 99)
- **Visualisations** : histogrammes, boxplots

#### [V] 1.4 - Analyser distributions des variables catégorielles
- Cardinalité par variable (faible < 10, moyenne 10-50, haute > 50)
- % de valeurs manquantes
- Identifier catégories rares (< 1% des observations)
- Déséquilibre des classes (imbalance ratio)
- **Focus spécial** : STRATUM (1316), OCOD (620), CNT (80)
- **Visualisations** : barplots, treemaps pour haute cardinalité

#### [V] 1.5 - Détecter variables redondantes / corrélées
- Calculer corrélations Spearman pour paires de variables ordinales
- Calculer Cramér's V pour paires de variables catégorielles
- Vérifier redondance CNT vs CNTRYID
- **Output** : Liste de variables à supprimer

#### [V] 1.6 - Analyser patterns de valeurs manquantes
- Matrice de corrélation des valeurs manquantes
- Identifier si missing est informatif (MCAR, MAR, MNAR)
- Décider stratégie d'imputation par variable

#### [V] 1.7 - Documenter conclusions EDA
- Créer rapport markdown avec décisions de preprocessing
- Lister variables à supprimer, à regrouper, à encoder
- Définir stratégies d'imputation par type de variable
- **Output** : `preprocessing/eda_conclusions.md`

#### [V] 1.8 - Analyser les colonnes d'après leurs noms et déduire ce qu'il faut supprimer / fusionner
- **Output** : `preprocessing/analysis/categorical_variables`, `preprocessing/analysis/ordinal_variables`

#### [V] 1.9 - Analyser des papiers de recherche sur ce sujet, en déduire les méthodes recommandées

#### [] 1.10 - Faire une synthèse des recommandations et mettre à jour cette to do list
- Synthèse à partir de `preprocessing/analysis/categorial_variables/RESUME_EXECUTIF_Analyse_Categorielles.md`, 
`preprocessing/analysis/ordinal_variables/RESUME_EXECUTIF_Analyse_Ordinales.md`, 
`preprocessing/analysis/analysis_from_pappers.md`et `preprocessing/01_eda_ordinal_categorical.ipynb`
- Mettre à jour la suite de cette To Do List.

---

### 🟡 PHASE 2 : CRÉATION DES CLASSES DE PREPROCESSING (dans `classes/`)
**Objectif** : Implémenter les classes basées sur les conclusions de l'EDA

#### [] 2.1 - Créer `classes/ordinal_preprocessor.py`
- Classe `OrdinalPreprocessor` avec méthodes pour les 96 variables ordinales
- Méthodes basées sur les conclusions de l'EDA (Phase 1)

#### [] 2.2 - Créer `classes/categorical_preprocessor.py`
- Classe `CategoricalPreprocessor` avec méthodes pour les 70 variables catégorielles
- Méthodes basées sur les conclusions de l'EDA (Phase 1)

---

### 🟢 PHASE 3 : GESTION DES VALEURS MANQUANTES (à implémenter dans les classes)

Stratégie à déterminer d'après la phase 1.

---

### 🟠 PHASE 4 : TRAITEMENT DES CATÉGORIES RARES (à implémenter dans les classes)

#### ☐ 4.1 - `group_rare_categories(df: pd.DataFrame, var: str, threshold: float = 0.01) -> pd.DataFrame`
**Objectif** : Regrouper catégories rares en "Other"
- Pour une variable catégorielle
- Regrouper modalités représentant < 1% (ou seuil) en "Other"
- Conserver mapping pour interprétabilité
- **Output** : DataFrame avec catégories regroupées

#### ☐ 4.2 - `reduce_stratum_dimensionality(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Réduire les 1316 strates en features exploitables
- Parser STRATUM pour extraire :
  - `stratum_location` : Urban / Rural
  - `stratum_region` : North / Center / South / etc.
  - `stratum_type` : Public / Private
  - `stratum_country` : Code pays (3 lettres)
- Supprimer STRATUM original
- **Output** : DataFrame avec 4 nouvelles variables + suppression STRATUM

#### ☐ 4.3 - `group_occupations_by_major_group(df: pd.DataFrame) -> pd.DataFrame`
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

#### ☐ 4.4 - `resolve_cnt_cntryid_redundancy(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Supprimer la redondance entre CNT et CNTRYID
- Vérifier corrélation parfaite
- Garder CNT (plus lisible : codes 3 lettres)
- Supprimer CNTRYID
- **Output** : DataFrame sans CNTRYID

---

### 🔵 PHASE 5 : ENCODAGE DES VARIABLES (à implémenter dans les classes)

#### ☐ 5.1 - `encode_ordinal_variables(df: pd.DataFrame, mapping_dict: dict = None) -> pd.DataFrame`
**Objectif** : Encoder les variables ordinales en préservant l'ordre
- Utiliser OrdinalEncoder de sklearn
- Créer mappings explicites pour échelles Likert, fréquences
- Exemple : {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
- Stocker encoders dans `self.encoders`
- **Output** : DataFrame avec ordinales encodées en int

#### ☐ 5.2 - `encode_binary_categorical(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Encoder variables catégorielles binaires
- Pour variables avec exactement 2 modalités (ex: Gender, OECD Yes/No)
- Encoder en 0/1 avec LabelEncoder
- **Output** : DataFrame avec binaires encodées

#### ☐ 5.3 - `onehot_encode_low_cardinality(df: pd.DataFrame, max_categories: int = 10) -> pd.DataFrame`
**Objectif** : One-Hot Encoding pour variables à faible cardinalité
- Pour variables catégorielles avec ≤ 10 modalités
- Utiliser pd.get_dummies ou OneHotEncoder
- Nommer colonnes : `var_name_category`
- **Output** : DataFrame avec colonnes one-hot créées

#### ☐ 5.4 - `target_encode_high_cardinality(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Target Encoding pour variables à haute cardinalité
- Pour variables avec > 10 modalités (CNT, langues, etc.)
- Encoder par moyenne de MathScore pour chaque catégorie
- Ajouter régularisation (smoothing) pour catégories rares
- Attention au data leakage : utiliser cross-validation
- **Output** : DataFrame avec target encoding appliqué

#### ☐ 5.5 - `frequency_encode_categorical(df: pd.DataFrame, vars_list: list) -> pd.DataFrame`
**Objectif** : Frequency Encoding (alternative au Target Encoding)
- Encoder par fréquence d'apparition de chaque catégorie
- Moins risqué que target encoding (pas de leakage)
- **Output** : DataFrame avec frequency encoding appliqué

---

### 🔴 PHASE 6 : VALIDATION ET CONTRÔLE QUALITÉ (à implémenter dans les classes)

#### ☐ 6.1 - `validate_no_missing_after_preprocessing(df: pd.DataFrame) -> bool`
**Objectif** : Vérifier qu'il n'y a plus de valeurs manquantes
- Compter les NaN restants
- Lever une exception si NaN détectés
- **Output** : True si OK, raise ValueError sinon

#### ☐ 6.2 - `validate_dtypes_after_encoding(df: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Vérifier les types de données après encodage
- Ordinales encodées → int ou float
- Catégorielles encodées → int ou float
- Pas de type 'object' sauf si voulu
- **Output** : DataFrame de validation avec [column, expected_dtype, actual_dtype, status]

#### ☐ 6.3 - `check_target_variable_unchanged(df_before: pd.DataFrame, df_after: pd.DataFrame) -> bool`
**Objectif** : Vérifier que MathScore n'a pas été modifié
- Comparer MathScore avant et après preprocessing
- Lever exception si différences détectées
- **Output** : True si identique, raise ValueError sinon

#### ☐ 6.4 - `generate_preprocessing_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict`
**Objectif** : Générer un rapport de preprocessing
- Nombre de variables avant/après
- Variables supprimées et raison
- Variables créées (one-hot, indicatrices missing)
- Statistiques d'encodage
- **Output** : Dict avec toutes les métadonnées

#### ☐ 6.5 - `detect_data_leakage_risk(df: pd.DataFrame) -> list`
**Objectif** : Détecter les risques de data leakage
- Identifier si target encoding fait sans CV
- Identifier si imputation utilise statistiques globales
- Identifier si normalisation faite sur tout le dataset
- **Output** : Liste des warnings de leakage potentiel

---

### 🟣 PHASE 7 : PIPELINE ET SAUVEGARDE (à implémenter dans les classes)

#### ☐ 7.1 - `create_preprocessing_pipeline(steps: list) -> Pipeline`
**Objectif** : Créer un pipeline sklearn réutilisable
- Enchaîner les transformations dans l'ordre
- Utiliser ColumnTransformer pour appliquer transformations par type
- **Output** : Pipeline sklearn fitted

#### ☐ 7.2 - `save_encoders_and_mappings(filepath: str) -> None`
**Objectif** : Sauvegarder les encoders pour réutilisation
- Pickler les OrdinalEncoder, LabelEncoder, OneHotEncoder
- Sauvegarder les mappings de référence
- Sauvegarder les listes de variables par type
- **Output** : Fichier .pkl

#### ☐ 7.3 - `export_preprocessed_data(df: pd.DataFrame, filepath: str) -> None`
**Objectif** : Exporter le dataset préprocessé
- Sauvegarder en CSV ou Parquet
- Inclure métadonnées dans un fichier séparé
- **Output** : Fichiers data + metadata

#### ☐ 7.4 - `transform_new_data(df_new: pd.DataFrame) -> pd.DataFrame`
**Objectif** : Appliquer le preprocessing à de nouvelles données
- Charger les encoders sauvegardés
- Appliquer les mêmes transformations
- Gérer les nouvelles catégories inconnues
- **Output** : DataFrame transformé


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

## 📊 LIVRABLES ATTENDUS

1. **Classe Python** : `OrdinalCategoricalPreprocessor` avec toutes les méthodes
2. **Notebook d'exemples** : Démonstration de chaque méthode
3. **Dataset preprocessé** : Fichier final prêt pour modélisation
4. **Documentation** : Rapport de preprocessing détaillé
5. **Encoders sauvegardés** : Fichiers .pkl pour réutilisation

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
- Prioriser la **vitesse** sur la qualité : seulement quelques heures pour cet exercice!
- **Documenter** toutes les décisions prises et les justifier