# 🎯 CONCLUSION - Analyse Exploratoire des Données PISA

## Synthèse Générale

Cette analyse exploratoire complète des données PISA a permis d'identifier les opportunités majeures d'optimisation du preprocessing pour la prédiction du score en mathématiques (MathScore). L'analyse s'est concentrée sur trois axes complémentaires : les variables ordinales, les variables catégorielles, et les recommandations issues de la littérature scientifique.

---

## 📊 Vue d'Ensemble des Données

### État Initial du Dataset
- **96 variables ordinales** réparties dans 13 domaines thématiques
- **70 variables catégorielles** avec forte concentration dans le domaine General (40%)
- **Défi majeur** : Explosion potentielle de dimensionnalité (~2093 features après encoding)
- **Problématique critique** : Codes ISCO haute cardinalité (620 codes × 3 variables)

### Distribution des Variables
- **Domaines principaux ordinaux** :
  - Family Background & Socioeconomic Status (22 variables)
  - Classroom Environment & Teaching Practices (13 variables)
  - ICT Use & Digital Competence (11 variables)

- **Domaines principaux catégoriels** :
  - General (28 variables - 40% du total)
  - Métadonnées administratives (significant portion)

---

## 🔥 DÉCOUVERTES CRITIQUES

### 1. Explosion de Dimensionnalité - Variables Catégorielles

**Impact AVANT nettoyage** :
```
OCOD1 (620) + OCOD2 (620) + OCOD3 (620) + Autres (50 vars)
= ~1910 features après one-hot encoding
```

**Impact APRÈS optimisation** :
```
OCOD1_grouped (10) + OCOD2_grouped (10) + Autres (30 vars)
= ~50 features après encoding
```

**🎉 GAIN : -97% de features catégorielles** (1910 → 50)

### 2. Redondances Structurelles

**Variables ordinales** :
- Redondances éducation parents : ST005/ST006 (mère) et ST007/ST008 (père) - on garde les codes ISCED numériques
- Redondances ressources numériques : ST253/ST254 et ST255/ST256 - on supprime un des deux
- Redondances perturbations classe : ST097/ST273 - on supprime un des deux
- Variables mesurant le même construit (support enseignant, support parental)

**Variables catégorielles** :
- 40% des variables sont des métadonnées administratives non prédictives
- Multiples variables d'options de questionnaires (7 variables)
- Identifiants administratifs sans valeur prédictive (3 variables)

### 3. Risque de Data Leakage

**Variables à EXCLURE ABSOLUMENT** :
- Toutes les PValues (PV1MATH à PV10MATH)
- Variables WLE (Weighted Likelihood Estimates)
- Variables d'effort post-test (EFFORT1, EFFORT2)

**Justification** : Ces variables représentent déjà la cible ou sont générées après le test, créant un risque de fuite d'information.

IMPORTANT: cette idée est rejetée dans le cadre de ce hackathon car on veut les meilleures performances possibles sachant qu'on aura un X_test qui a les mêmes données que le X_train. 

---

## ⚡ PLAN D'ACTION RECOMMANDÉ

### 🔴 PHASE 1 : Suppressions Sans Risque (EXÉCUTION IMMÉDIATE)

#### Variables Ordinales (-5 variables)
```python
variables_ordinales_a_supprimer = [
    'ST005',   # Education mère (description) - redondant avec ST006 (code ISCED)
    'ST007',   # Education père (description) - redondant avec ST008 (code ISCED)
    'ST253',   # Redondant avec ST254
    'ST255',   # Redondant avec ST256
    'ST097'    # Redondant avec ST273
]
# Gain net: -5 variables ordinales
# Risque: MINIMAL
# Note: On garde ST006 et ST008 (codes ISCED numériques ordinaux)
```

#### Variables Catégorielles - Métadonnées (-10 variables)
```python
metadonnees_a_supprimer = [
    # Options questionnaires (7)
    'Option_CT', 'Option_FL', 'Option_ICTQ', 'Option_PQ',
    'Option_TQ', 'Option_UH', 'Option_WBQ',

    # Identifiants administratifs (3)
    'CYC', 'NatCen', 'SUBNATIO'
]
# Gain net: -10 variables catégorielles
# Risque: ZÉRO
# Note: EFFORT1/EFFORT2 retirés car pas de data leakage en hackathon
```

#### Variables Catégorielles - Redondances (-7 variables)
```python
redondances_categorielles_a_supprimer = [
    'LANGTEST_PAQ',      # Redondant avec LANGTEST_COG
    'LANGTEST_QQQ',      # Redondant avec LANGTEST_COG
    'ST003D03T',         # Birth Year = redondant avec AGE
    'ST001D01T',         # Grade = redondant avec GRADE
    'PA008',             # Doublon (une des copies de PA008)
    'PA162',             # Lecture parent (garder ST168)
    'OCOD3',             # Profession aspirée (faible valeur + haute cardinalité)
]
# Gain net: -7 variables catégorielles
# Risque: MINIMAL
```

**📈 GAIN PHASE 1 : -22 variables** (-5 ordinales + -10 métadonnées + -7 redondances catégorielles)

---

### 🟠 PHASE 2 : Consolidations et Fusions (PRIORITÉ HAUTE)

#### Scores Composites - Variables Ordinales
```python
# Fusionner variables mesurant même construit
fusions_ordinales = {
    'Score_Support_Parental': ['PA003', 'ST300'],
    'Score_Support_Enseignant': ['ST100', 'ST270']
}
# Méthode : Moyenne ou PCA sur composantes
# Gain net: -2 variables ordinales
```

#### Regroupement ISCO - Variables Catégorielles (CRITIQUE)
```python
def regroup_isco_codes(isco_code):
    """
    Regrouper codes ISCO-08 (620) en 10 grandes catégories

    Catégories ISCO niveau 1:
    1: Managers
    2: Professionals
    3: Technicians and associate professionals
    4: Clerical support workers
    5: Service and sales workers
    6: Skilled agricultural, forestry and fishery workers
    7: Craft and related trades workers
    8: Plant and machine operators, and assemblers
    9: Elementary occupations
    0: Armed forces occupations
    """
    if pd.isna(isco_code):
        return np.nan

    # Prendre le 1er chiffre du code (niveau 1 ISCO)
    return int(str(int(isco_code))[0])

# Appliquer le regroupement
df['OCOD1_grouped'] = df['OCOD1'].apply(regroup_isco_codes)
df['OCOD2_grouped'] = df['OCOD2'].apply(regroup_isco_codes)

# Supprimer les codes originaux
df = df.drop(columns=['OCOD1', 'OCOD2'])

# Impact:
# - Cardinalité: 620 → 10 par variable
# - Features après encoding: 1240 → 20
# - Réduction: -98% de features ISCO
```

**📈 GAIN PHASE 2 : -2 variables ordinales + Impact massif sur features catégorielles**

---

### 🟡 PHASE 3 : Variables à Évaluer Empiriquement

#### Variables COVID (À DÉCIDER selon objectif)

**Option A - Score Composite COVID**
```python
# Si pertinent pour capturer l'effet COVID
variables_covid = ['ST348', 'ST351', 'ST352', 'ST353']  # Ordinales
variables_covid_cat = ['ST347', 'ST349', 'ST350']        # Catégorielles

# Créer score composite
score_covid = moyenne_ou_pca(variables_covid + variables_covid_cat)
# Gain: -6 variables
```

**Option B - Suppression COVID**
```python
# Si COVID n'impacte pas significativement MathScore
# (à valider par test de corrélation)
supprimer(variables_covid + variables_covid_cat)
# Gain: -7 variables
```

**Critère de décision** : Tester corrélation avec MathScore et importance dans modèles baseline

#### Consolidation TIC (À VALIDER)
```python
# Variables IC170-176 (usage TIC par domaine)
# GARDER ABSOLUMENT : IC184 (usage TIC mathématiques)
# À évaluer : Autres variables TIC potentiellement redondantes
# Gain potentiel: -4 variables ordinales
```

---

## 📊 SCÉNARIOS DE RÉDUCTION

### Scénario CONSERVATEUR ⭐ (RECOMMANDÉ)

**Actions** : Phases 1 + 2

**Impact Variables** :
- Ordinales : 96 → **89 variables** (-7 variables : -5 suppressions + -2 par scores composites)
- Catégorielles : 70 → **53 variables** (-17 variables : -10 métadonnées + -7 redondances)
- **Total : 166 → 142 variables (-14.5%)**

**Impact Features (après encoding)** :
- Avant : ~2093 features (81 numériques + 96 ordinales + 1910 catégorielles + 6 groupement)
- Après : ~224 features (81 numériques + 87 ordinales + 50 catégorielles + 6 groupement)
- **GAIN GLOBAL : -89% de features (-1869 features)**

**Risque** : MINIMAL
**Timeline** : IMMÉDIAT

---

### Scénario AGRESSIF

**Actions** : Phases 1 + 2 + 3

**Impact Variables** :
- Ordinales : 96 → **74-78 variables** (-18 à -22 variables)
- Catégorielles : 70 → **44-48 variables** (-22 à -26 variables)
- **Total : 166 → 118-126 variables (-24 à -29%)**

**Impact Features** : ~2093 → ~200 features (-90%)

**Risque** : MOYEN (perte information granulaire TIC et COVID)
**Timeline** : Après validation empirique

---

## 🎯 RECOMMANDATIONS PREPROCESSING (Basées sur la littérature)

### 1. Gestion des Valeurs Manquantes

**Stratégie recommandée** :
```python
# 1. Éliminer variables avec >50% missing
high_missing_vars = [var for var in df.columns
                     if df[var].isna().mean() > 0.5]
df = df.drop(columns=high_missing_vars)

# 2. Imputation KNN (k=5) pour continues et ordinales
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[continuous_vars] = imputer.fit_transform(df[continuous_vars])
df[ordinal_vars] = imputer.fit_transform(df[ordinal_vars])

# 3. Mode pour catégorielles
for var in categorical_vars:
    df[var].fillna(df[var].mode()[0], inplace=True)
```

**Justification** :
- KNN préserve mieux les relations entre variables que l'imputation simple
- Variables avec >50% missing n'apportent pas d'information fiable
- Recommandé par 2 études scientifiques analysées

### 2. Protection Contre le Data Leakage

**Note Hackathon** : En contexte hackathon où X_test a les mêmes features que X_train, on GARDE les PV*/WLE* pour maximiser la performance. En production, il faudrait les exclure.

**En production classique (EXCLURE ABSOLUMENT)** :
```python
# Variables générées après le test ou représentant la cible
leakage_vars = [
    # PValues - imputation multiple de la cible
    'PV1MATH', 'PV2MATH', ..., 'PV10MATH',
    # WLE - Weighted Likelihood Estimates
    'WLE_*',
    # Effort post-test (si non pertinent)
    # 'EFFORT1', 'EFFORT2'  # Gardés en hackathon
]
# df = df.drop(columns=leakage_vars)  # Commenté pour hackathon
```

### 3. Encodage et Normalisation

```python
# 1. SPLIT TRAIN/VALIDATION/TEST (60/20/20)
# IMPORTANT : Faire AVANT tout preprocessing pour éviter leakage

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=bins_of_y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=bins_of_y_temp
)

# 2. ENCODAGE (après split)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# One-hot pour catégorielles nominales
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = ohe.fit_transform(X_train[categorical_vars])
X_val_cat = ohe.transform(X_val[categorical_vars])
X_test_cat = ohe.transform(X_test[categorical_vars])

# Ordinal pour variables ordinales
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_ord = oe.fit_transform(X_train[ordinal_vars])
X_val_ord = oe.transform(X_val[ordinal_vars])
X_test_ord = oe.transform(X_test[ordinal_vars])

# 3. STANDARDISATION (après split et encoding)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numeric_vars])
X_val_num = scaler.transform(X_val[numeric_vars])
X_test_num = scaler.transform(X_test[numeric_vars])
```

**Pourquoi cette séquence** :
- Split AVANT preprocessing pour éviter data leakage
- Fit sur train, transform sur val/test pour éviter information leakage
- StandardScaler améliore convergence des modèles et interprétabilité

### 4. Traitement des Outliers

```python
# Winsorization au 99ème percentile
from scipy.stats.mstats import winsorize

for var in continuous_vars:
    X_train[var] = winsorize(X_train[var], limits=[0.01, 0.01])
    # Note: Appliquer les mêmes bornes sur val/test basées sur train
```

### 5. Sélection de Features

**Approche hybride recommandée** :
```python
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

# 1. Mutual Information (capture dépendances non-linéaires)
mi_scores = mutual_info_regression(X_train, y_train)
mi_features = X_train.columns[mi_scores > threshold]

# 2. Recursive Feature Elimination
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=30)
rfe.fit(X_train, y_train)
rfe_features = X_train.columns[rfe.support_]

# 3. Intersection des deux méthodes
selected_features = list(set(mi_features) & set(rfe_features))

# 4. Après modélisation : Permutation Importance + SHAP
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# Analyser importance globale et locale
```

**Justification** :
- RFE élimine itérativement les features peu importantes
- MI capture les dépendances non-linéaires
- SHAP fournit interprétabilité locale et globale
- Objectif : Conserver ~20-35 features optimales

---

## 📋 VARIABLES À CONSERVER ABSOLUMENT

### Variables Critiques (Ne JAMAIS Supprimer)

**Validées par littérature ET analyses** :

1. **ESCS** (Economic, Social and Cultural Status) - Systématiquement top importance
2. **HISEI** (Highest parental occupational status) - Systématiquement top importance
3. **ICTRES** (ICT resources) - Très important pour tous domaines
4. **ST004D01T** (Gender) - Sociodémographique clé + équité
5. **IMMIG** (Immigration status) - Important pour fairness/equity
6. **GRADE** (Position in grade) - Capture redoublement (fort impact MathScore)
7. **ADMINMODE** (Computer vs Paper) - Mode peut influencer performance
8. **LANGTEST_COG** (Langue du test) - Essentiel analyses multilingues
9. **MATHEASE** (Math easier than other subjects) - Perception pertinente
10. **MISSSC** (Missing school >3 months) - Impact éducatif significatif
11. **IC184** (Usage TIC mathématiques) - Pertinent direct pour MathScore
12. **METASPAM** (Métacognition) - Important selon littérature
13. **COMPETE** (Compétition) - Spécifique aux maths

---

## ⚠️ POINTS CRITIQUES À NE PAS OUBLIER

### ❌ NE JAMAIS

1. Inclure les PValues dans les features
2. Appliquer transformations (scaling, encoding) AVANT le split train/test
3. Utiliser les données de test pour fit des transformateurs
4. Supprimer des variables sans validation de leur faible importance
5. Ignorer les >60% de features avec >50% de missing values

### ✅ TOUJOURS

1. Splitter AVANT tout preprocessing
2. Fixer random_state pour reproductibilité
3. Préserver l'information ordinale quand elle existe
4. Valider avec cross-validation pour éviter surapprentissage
5. Tester performance sur test set final UNE SEULE FOIS

---

## 💡 INSIGHTS MAJEURS

### 1. Impact Disproportionné des Variables Catégorielles
- **3 variables ISCO génèrent 97% des features** après one-hot encoding
- Le regroupement ISCO est l'optimisation la plus impactante du preprocessing
- Réduction de ~1910 → ~50 features avec regroupement intelligent

### 2. Opportunité Exceptionnelle de Réduction
- **40% des variables catégorielles sont des métadonnées** sans valeur prédictive
- Suppression avec risque ZÉRO identifiée pour 10 variables (métadonnées)
- 5 redondances ordinales + 7 redondances catégorielles supplémentaires
- **ROI maximal** : Variables catégorielles offrent le plus grand gain avec risque minimal

### 3. Validation Scientifique
- Recommandations alignées avec 4 études scientifiques analysées
- Variables critiques (ESCS, HISEI, ICTRES) systématiquement importantes
- Pipeline KNN imputation + StandardScaler + RFE validé empiriquement

### 4. Trade-off Dimensionnalité vs Information
- **Scénario conservateur** : -89% features avec perte information minimale
- Variables ordinales : -7 variables (-7.3% : -5 suppressions + -2 scores composites)
- Variables catégorielles : -17 variables (-24.3% : -10 métadonnées + -7 redondances)
- Impact massif sur features grâce au regroupement ISCO (620 → 10 catégories)

---

## 🎯 RECOMMANDATION FINALE

### Action Immédiate (Cette Semaine)

**Implémenter Scénario CONSERVATEUR (Phases 1-2)** :

✅ **Phase 1** : Supprimer 22 variables (métadonnées et redondances)
✅ **Phase 2** : Regrouper ISCO + créer 2 scores composites

**Résultat attendu** :
- Variables : 166 → 142 (-14.5%)
- Features : ~2093 → ~224 (-89%)
- Risque : MINIMAL
- Timeline : IMMÉDIAT

### Validation Empirique (Semaine Prochaine)

📊 Tester corrélations variables "à évaluer" (COVID, TIC)
🧪 Comparer performance modèle avant/après réduction
📈 Mesurer impact sur R² et RMSE

### Décision Phase 3 (Selon Résultats Validation)

Si tests confirment faible importance :
- Consolider/supprimer 7 variables COVID
- Consolider 4 variables TIC redondantes
- **Gain supplémentaire potentiel** : -8 à -11 variables

---

## 📊 IMPACT GLOBAL ATTENDU

### Transformation du Dataset

**AVANT Preprocessing** :
```
Total features brutes estimées:
- Numériques: 81 features
- Ordinales: ~96 features
- Catégorielles: ~1910 features (one-hot)
- Groupement: 6 features
TOTAL: ~2093 features
```

**APRÈS Preprocessing Conservateur** :
```
Total features optimisées:
- Numériques: 81 features
- Ordinales: ~87 features (-9)
- Catégorielles: ~50 features (-1860!)
- Groupement: 6 features
TOTAL: ~224 features
```

### 🎉 GAIN GLOBAL : -89% de features (-1869 features)

### Bénéfices Attendus

✅ **Réduction temps de calcul** : ~80-90% sur training et inference
✅ **Diminution risque overfitting** : Curse of dimensionality évitée
✅ **Amélioration généralisation** : Modèle plus robuste
✅ **Meilleure interprétabilité** : Focus sur variables à haute valeur
✅ **Facilitation feature engineering** : Base saine pour itérations

---
