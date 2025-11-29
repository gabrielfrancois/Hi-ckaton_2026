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

#### [V] 1.10 - Faire une synthèse des recommandations et mettre à jour cette to do list
- Synthèse à partir de `preprocessing/analysis/categorial_variables/RESUME_EXECUTIF_Analyse_Categorielles.md`,
`preprocessing/analysis/ordinal_variables/RESUME_EXECUTIF_Analyse_Ordinales.md`,
`preprocessing/analysis/analysis_from_pappers.md`et `preprocessing/01_eda_ordinal_categorical.ipynb`
- **Output** : `preprocessing/analysis/conclusion_eda.md`
- Mettre à jour la suite de cette To Do List.

---

### 🟡 PHASE 2 : CRÉATION DES CLASSES DE PREPROCESSING
**Objectif** : Implémenter le scénario CONSERVATEUR (Phases 1-2) avec 2 classes OOP

#### [] 2.1 - Créer classe `OrdinalPreprocessor` dans `preprocessing/classes/ordinal_preprocessor.py`
**Objectif** : Classe pour gérer toutes les transformations des variables ordinales

**Méthodes à implémenter** :

```python
class OrdinalPreprocessor:
    def __init__(self):
        self.ordinal_vars = []  # Liste variables ordinales
        self.variables_to_drop = []
        self.composite_scores = {}
        self.encoders = {}
        self.scaler = None

    # Phase 1 : Nettoyage
    def drop_redundant_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprimer 5 variables ordinales redondantes"""
        # ST005, ST007, ST253, ST255, ST097

    # Phase 2 : Scores composites
    def create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer scores composites pour variables mesurant même construit"""
        # Score_Support_Parental = moyenne(PA003, ST300)
        # Score_Support_Enseignant = moyenne(ST100, ST270)

    # Phase 3 : Imputation
    def impute_knn(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                   df_test: pd.DataFrame, k: int = 5) -> tuple:
        """Imputer valeurs manquantes avec KNN (fit sur train)"""

    # Phase 4 : Traitement outliers
    def winsorize_outliers(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                          df_test: pd.DataFrame, limits: list = [0.01, 0.01]) -> tuple:
        """Winsorization au 99ème percentile"""

    # Phase 5 : Encodage
    def encode_ordinal_variables(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                                df_test: pd.DataFrame) -> tuple:
        """Encoder variables ordinales en préservant l'ordre (fit sur train)"""

    # Phase 6 : Standardisation
    def standardize_variables(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                             df_test: pd.DataFrame) -> tuple:
        """Standardiser variables ordinales (fit sur train)"""

    # Utils
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline complet pour train"""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appliquer transformations sur val/test"""

    def save(self, filepath: str):
        """Sauvegarder preprocessor"""

    @staticmethod
    def load(filepath: str):
        """Charger preprocessor sauvegardé"""
```

**Sous-tâches** :
- Implémenter les 9 méthodes ci-dessus
- Documenter chaque méthode avec docstrings
- Gérer les valeurs manquantes lors des scores composites
- Stocker tous les transformers (encoders, scaler) comme attributs
- **Gain** : -7 variables ordinales (-5 suppressions + -2 par scores composites)

#### [] 2.2 - Créer classe `CategoricalPreprocessor` dans `preprocessing/classes/categorical_preprocessor.py`
**Objectif** : Classe pour gérer toutes les transformations des variables catégorielles

**Méthodes à implémenter** :

```python
class CategoricalPreprocessor:
    def __init__(self):
        self.categorical_vars = []  # Liste variables catégorielles
        self.variables_to_drop = []
        self.isco_mapping = {}
        self.rare_categories_mapping = {}
        self.binary_encoders = {}
        self.onehot_encoder = None
        self.frequency_encoders = {}

    # Phase 1 : Nettoyage
    def drop_metadata_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprimer 10 métadonnées catégorielles (risque ZÉRO)"""
        # Option_CT, Option_FL, Option_ICTQ, Option_PQ, Option_TQ,
        # Option_UH, Option_WBQ, CYC, NatCen, SUBNATIO

    def drop_redundant_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprimer 7 redondances catégorielles"""
        # LANGTEST_PAQ, LANGTEST_QQQ, ST003D03T, ST001D01T,
        # PA008, PA162, OCOD3

    # Phase 2 : Regroupement ISCO (CRITIQUE)
    def group_isco_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regrouper codes ISCO (620 → 10 catégories)"""
        # OCOD1 → OCOD1_grouped (10 catégories)
        # OCOD2 → OCOD2_grouped (10 catégories)
        # Impact : -1240 features potentielles

    # Phase 3 : Imputation
    def impute_mode(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                    df_test: pd.DataFrame) -> tuple:
        """Imputer avec mode (calculé sur train)"""

    # Phase 4 : Catégories rares
    def group_rare_categories(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                              df_test: pd.DataFrame, threshold: float = 0.01) -> tuple:
        """Regrouper catégories < 1% en 'Other' (fit sur train)"""

    # Phase 5 : Encodage
    def encode_binary_variables(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                                df_test: pd.DataFrame) -> tuple:
        """Encoder variables binaires en 0/1 (fit sur train)"""

    def onehot_encode_low_cardinality(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                                      df_test: pd.DataFrame, max_categories: int = 10) -> tuple:
        """One-Hot encoding pour cardinalité ≤10 (fit sur train)"""

    def frequency_encode_high_cardinality(self, df_train: pd.DataFrame, df_val: pd.DataFrame,
                                         df_test: pd.DataFrame) -> tuple:
        """Frequency encoding pour cardinalité >10 (fit sur train)"""

    # Utils
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline complet pour train"""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appliquer transformations sur val/test"""

    def save(self, filepath: str):
        """Sauvegarder preprocessor"""

    @staticmethod
    def load(filepath: str):
        """Charger preprocessor sauvegardé"""
```

**Sous-tâches** :
- Implémenter les 12 méthodes ci-dessus
- Documenter chaque méthode avec docstrings
- Implémenter fonction `regroup_isco_codes()` pour extraire 1er chiffre
- Stocker tous les encoders/mappings comme attributs
- **Gain** : -17 variables catégorielles (-10 métadonnées + -7 redondances)

#### [] 2.3 - Créer fonctions utilitaires `preprocessing/utils/preprocessing_utils.py`
**Objectif** : Fonctions auxiliaires pour orchestrer les 2 preprocessors

**Fonctions à implémenter** :

```python
def remove_high_missing_vars(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Supprimer variables avec >50% missing"""
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > threshold].index.tolist()
    print(f"Suppression de {len(high_missing)} variables avec >{threshold*100}% missing")
    return df.drop(columns=high_missing)

def split_train_val_test(df: pd.DataFrame, target: str = 'MathScore',
                        test_size: float = 0.2, val_size: float = 0.2,
                        random_state: int = 42) -> tuple:
    """Split stratifié 60/20/20 sur bins de target"""
    # Créer bins pour stratification
    # Retourner X_train, X_val, X_test, y_train, y_val, y_test

def validate_preprocessing(df_before: pd.DataFrame, df_after: pd.DataFrame,
                          target: str = 'MathScore') -> dict:
    """Valider preprocessing (no missing, dtypes, target unchanged)"""
    # Vérifier 0 NaN
    # Vérifier target identique
    # Retourner rapport validation

def generate_preprocessing_report(df_before: pd.DataFrame, df_after: pd.DataFrame,
                                 ordinal_prep, categorical_prep) -> dict:
    """Générer rapport preprocessing complet (markdown + JSON)"""
    # Variables supprimées
    # Variables créées
    # Statistiques imputation
    # Retourner dict avec toutes les métadonnées
```

**Sous-tâches** :
- Implémenter les 4 fonctions ci-dessus
- Documenter avec docstrings
- Gérer stratification sur bins de MathScore
- **Output** : Module utils avec fonctions helper

#### [] 2.4 - Créer tests unitaires `preprocessing/tests/test_preprocessors.py`
**Objectif** : Tester chaque méthode des classes

**Sous-tâches** :
- Tester OrdinalPreprocessor (drop, composite, encode, etc.)
- Tester CategoricalPreprocessor (drop, ISCO, encode, etc.)
- Tester PISAPreprocessor (pipeline complet)
- Tester que MathScore n'est jamais modifié
- Tester absence de data leakage (fit/transform séparés)
- **Output** : Suite de tests avec pytest

---

### 🟢 PHASE 3 : FEATURE SELECTION ET VALIDATION

#### [] 6.1 - Implémenter feature selection hybride `preprocessing/scripts/feature_selection.py`
**Objectif** : Sélectionner ~20-35 features optimales (recommandation littérature)

**Sous-tâches** :
- Calculer Mutual Information sur train set
  ```python
  from sklearn.feature_selection import mutual_info_regression
  mi_scores = mutual_info_regression(X_train, y_train)
  ```
- Implémenter Recursive Feature Elimination avec RandomForest
  ```python
  from sklearn.feature_selection import RFE
  rfe = RFE(estimator=rf, n_features_to_select=30)
  ```
- Créer intersection des features sélectionnées par les 2 méthodes
- **Output** : Liste de features sélectionnées + scores d'importance

#### [] 3.2 - Créer notebook de validation `preprocessing/02_validation_preprocessing.ipynb`
**Objectif** : Valider le preprocessing complet et analyser résultats

**Pipeline d'exécution** :
```python
from classes.ordinal_preprocessor import OrdinalPreprocessor
from classes.categorical_preprocessor import CategoricalPreprocessor
from utils.preprocessing_utils import *

# 1. Charger données
df = pd.read_csv('data/X_train.csv')

# 2. Remove high missing
df = remove_high_missing_vars(df, threshold=0.5)

# 3. Appliquer nettoyage (avant split)
ordinal_prep = OrdinalPreprocessor()
categorical_prep = CategoricalPreprocessor()

df = ordinal_prep.drop_redundant_variables(df)
df = categorical_prep.drop_metadata_variables(df)
df = categorical_prep.drop_redundant_variables(df)
df = categorical_prep.group_isco_codes(df)
df = ordinal_prep.create_composite_scores(df)

# 4. Split train/val/test
X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(df)

# 5. Appliquer transformations (fit sur train)
X_train, X_val, X_test = ordinal_prep.impute_knn(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.impute_mode(X_train, X_val, X_test)
X_train, X_val, X_test = ordinal_prep.winsorize_outliers(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.group_rare_categories(X_train, X_val, X_test)
X_train, X_val, X_test = ordinal_prep.encode_ordinal_variables(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.encode_binary_variables(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.onehot_encode_low_cardinality(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.frequency_encode_high_cardinality(X_train, X_val, X_test)
X_train, X_val, X_test = ordinal_prep.standardize_variables(X_train, X_val, X_test)

# 6. Validation
validation_report = validate_preprocessing(df, X_train, target='MathScore')
preprocessing_report = generate_preprocessing_report(df, X_train, ordinal_prep, categorical_prep)
```

**Sous-tâches** :
- Exécuter pipeline complet ci-dessus
- Valider absence NaN après preprocessing
- Vérifier MathScore non modifié
- Visualiser distributions avant/après
- Comparer statistiques descriptives
- **Output** : Notebook validation + rapport

---

### 🟣 PHASE 4 : UTILISATION ET EXPORT

#### [] 4.1 - Créer script d'utilisation `preprocessing/run_preprocessing.py`
**Objectif** : Script principal pour lancer le preprocessing complet

**Sous-tâches** :
```python
from classes.ordinal_preprocessor import OrdinalPreprocessor
from classes.categorical_preprocessor import CategoricalPreprocessor
from utils.preprocessing_utils import *
import pandas as pd

# Charger données
df = pd.read_csv('data/X_train.csv')
df_before = df.copy()

# 1. Remove high missing
df = remove_high_missing_vars(df, threshold=0.5)

# 2. Nettoyage (avant split)
ordinal_prep = OrdinalPreprocessor()
categorical_prep = CategoricalPreprocessor()

df = ordinal_prep.drop_redundant_variables(df)
df = categorical_prep.drop_metadata_variables(df)
df = categorical_prep.drop_redundant_variables(df)
df = categorical_prep.group_isco_codes(df)
df = ordinal_prep.create_composite_scores(df)

# 3. Split
X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(df)

# 4. Transformations (fit sur train)
X_train, X_val, X_test = ordinal_prep.impute_knn(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.impute_mode(X_train, X_val, X_test)
X_train, X_val, X_test = ordinal_prep.winsorize_outliers(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.group_rare_categories(X_train, X_val, X_test)
X_train, X_val, X_test = ordinal_prep.encode_ordinal_variables(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.encode_binary_variables(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.onehot_encode_low_cardinality(X_train, X_val, X_test)
X_train, X_val, X_test = categorical_prep.frequency_encode_high_cardinality(X_train, X_val, X_test)
X_train, X_val, X_test = ordinal_prep.standardize_variables(X_train, X_val, X_test)

# 5. Sauvegarder preprocessors
ordinal_prep.save('models/ordinal_preprocessor.pkl')
categorical_prep.save('models/categorical_preprocessor.pkl')

# 6. Sauvegarder datasets
X_train.to_csv('data/processed/X_train_preprocessed.csv', index=False)
X_val.to_csv('data/processed/X_val_preprocessed.csv', index=False)
X_test.to_csv('data/processed/X_test_preprocessed.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_val.to_csv('data/processed/y_val.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

# 7. Générer rapport
report = generate_preprocessing_report(df_before, X_train, ordinal_prep, categorical_prep)
```

#### [] 4.2 - Créer notebook démo `preprocessing/03_demo_preprocessing.ipynb`
**Objectif** : Démonstration complète du preprocessing

**Sous-tâches** :
- Charger données brutes
- Montrer étape par étape chaque transformation
- Visualiser impact de chaque étape
- Afficher statistiques avant/après
- Sauvegarder résultats finaux
- **Output** : Notebook démo commenté


---

## 🎯 INSIGHTS CLÉS DE L'ANALYSE EDA (Synthèse Phase 1.10)

### 💡 Découvertes Majeures

1. **Explosion de Dimensionnalité Évitée**
   - **AVANT** : ~2093 features après one-hot encoding
   - **APRÈS** : ~224 features avec scénario conservateur
   - **GAIN** : -89% de features (-1869 features)
   - **Impact principal** : Regroupement ISCO (620 → 10 catégories) = -97% features catégorielles

2. **Métadonnées Non Prédictives**
   - 40% des variables catégorielles sont des métadonnées administratives
   - Suppression de 12 variables avec risque ZÉRO
   - Variables : Options questionnaires, identifiants admin, effort post-test

3. **Redondances Structurelles Identifiées**
   - 7 variables ordinales redondantes (doublons ISCED, ressources numériques)
   - 7 variables catégorielles redondantes (langues, dates, professions)
   - Opportunité de créer 2 scores composites (support parental, support enseignant)

4. **Scénario CONSERVATEUR Recommandé**
   - Phases 1-2 : Réduction 166 → 138 variables (-16.9%)
   - Risque MINIMAL validé par analyses multiples
   - Timeline : IMMÉDIAT (cette semaine)
   - Validation empirique Phase 3 (COVID, TIC) : Semaine prochaine

5. **Variables Critiques à Conserver** (Validées littérature + analyses)
   - ESCS, HISEI, ICTRES : Top importance systématique
   - Gender, IMMIG, GRADE : Sociodémographiques essentiels
   - IC184, METASPAM, COMPETE : Spécifiques mathématiques

6. **Pipeline Validé par Littérature**
   - KNN imputation (k=5) pour continues/ordinales
   - Mode imputation pour catégorielles
   - StandardScaler après encoding
   - Feature selection hybride (RFE + MI) → ~20-35 features
   - Cross-validation 5-fold pour validation

### 📊 Ordre des Transformations (CRITIQUE pour éviter Data Leakage)

**SÉQUENCE STRICTE À RESPECTER** :
1. Remove PV*/WLE* (avant tout)
2. Remove high missing >50%
3. Clean variables (Phases 1-2)
4. **SPLIT TRAIN/VAL/TEST** ← Point critique
5. Imputation (fit train, transform val/test)
6. Outliers treatment (fit train, transform val/test)
7. Encoding (fit train, transform val/test)
8. Standardization (fit train, transform val/test)
9. Feature selection (train only)

### 🎓 Recommandations Issues de 4 Études Scientifiques
- **Imputation** : KNN (k=5) > Simple mean/median
- **Encoding haute cardinalité** : Frequency > Target (évite leakage)
- **Standardisation** : Obligatoire pour convergence modèles
- **Feature selection** : RFE + MI > univarié seul
- **Validation** : Stratified 5-fold CV
- **Split** : 60/20/20 avec stratification sur target bins

### ⚡ Quick Wins Identifiés
1. **Gain immédiat** : Supprimer 22 variables métadonnées/redondances (-13.3% variables)
   - 5 ordinales (ST005, ST007, ST253, ST255, ST097)
   - 10 métadonnées catégorielles (Options + identifiants admin)
   - 7 redondances catégorielles (langues, dates, professions)
2. **Gain massif** : Regrouper ISCO (-1860 features potentielles → -97% features catégorielles)
3. **Gain consolidation** : 2 scores composites (-2 variables ordinales)
4. **Total Phase 1-2** : 166 → 142 variables (-14.5%), ~2093 → ~224 features (-89%)
5. **Note Hackathon** : PV*/WLE* gardés pour performance maximale

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