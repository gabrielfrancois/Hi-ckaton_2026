Je vais analyser les articles scientifiques de la base de connaissance pour identifier les meilleures pratiques de preprocessing pour prédire le score PISA.Basé sur l'analyse des articles scientifiques de la base de connaissance, voici mes **recommandations synthétiques pour le preprocessing des données PISA** :

## 📊 SYNTHÈSE DES RECOMMANDATIONS

### 1. **GESTION DES VALEURS MANQUANTES** 

**Quoi faire :**
- **Imputation KNN (k=5)** pour les variables continues et ordinales (recommandé par 2 études)
- **Mode/valeur la plus fréquente** pour les variables catégorielles
- **Éliminer les variables** avec >50% de valeurs manquantes

**Pourquoi :**
- KNN préserve mieux les relations entre variables que l'imputation simple
- Évite d'introduire du bruit avec des features artificiellement remplies
- Les variables avec >50% de missing n'apportent pas d'information fiable

### 2. **PROTECTION CONTRE LE DATA LEAKAGE**

**Quoi faire :**
- **EXCLURE ABSOLUMENT** tous les PValues (PV1MATH à PV10MATH, etc.)
- Exclure les variables WLE (Weighted Likelihood Estimates)
- Ne conserver qu'UN seul score moyen par domaine si nécessaire

**Pourquoi :**
- Les PValues sont générés par imputation multiple et représentent déjà la cible
- Les inclure donne une précision artificiellement élevée mais inutilisable
- Risque de fuite d'information entre train/test

### 3. **NORMALISATION ET STANDARDISATION**

**Quoi faire :**
- **Min-Max Scaling [0,1]** pour les algorithmes sensibles à la magnitude (SVM, réseaux de neurones)
- **StandardScaler (mean=0, std=1)** pour les variables continues avant modélisation
- Appliquer après le split train/test pour éviter le data leakage

**Pourquoi :**
- Améliore la convergence des modèles
- Met toutes les features sur une échelle comparable
- Essentiel pour l'interprétabilité des coefficients

### 4. **ENCODAGE DES VARIABLES**

**Quoi faire :**
- **One-hot encoding** pour les variables catégorielles nominales
- **Ordinal encoding (rangs entiers)** pour les variables ordinales (ex: nombre de livres à la maison)
- **Variables binaires** en 0/1

**Pourquoi :**
- One-hot évite d'imposer un ordre artificiel sur les catégories nominales
- L'ordinal encoding préserve l'information d'ordre naturel
- Compatible avec tous les algorithmes ML

### 5. **SÉLECTION DE FEATURES**

**Quoi faire :**
- **Approche hybride** : Recursive Feature Elimination (RFE) + Mutual Information (MI)
- **Permutation Importance** après modélisation pour identifier les top features
- Utiliser **SHAP values** pour l'analyse d'importance
- Conserver ~20-35 features optimales selon les analyses

**Pourquoi :**
- RFE élimine itérativement les features peu importantes
- MI capture les dépendances non-linéaires
- SHAP fournit une interprétabilité locale et globale
- Réduit le surapprentissage et améliore la généralisation

### 6. **GESTION DE LA DIMENSIONNALITÉ**

**Quoi faire :**
- Envisager **UMAP** pour visualisation (pas nécessairement pour modélisation)
- Filtrer les features basé sur l'importance permutée
- Éliminer les variables redondantes (multicolinéarité)

**Pourquoi :**
- 308 variables → trop de dimensions, risque de curse of dimensionality
- Les modèles tree-based gèrent bien la multicolinéarité, mais pas les modèles linéaires
- La réduction dimensionnelle améliore l'efficacité computationnelle

### 7. **TRAITEMENT DES VALEURS EXTRÊMES**

**Quoi faire :**
- **Winsorization** au 99ème percentile pour les variables continues (ex: temps d'étude)
- Identifier et traiter les valeurs aberrantes irréalistes

**Pourquoi :**
- Certains élèves reportent des valeurs irréalistes
- Limite l'influence excessive des outliers

### 8. **SPLIT DES DONNÉES**

**Quoi faire :**
- **60% train / 20% validation / 20% test** (standard)
- **Stratified K-Fold Cross-Validation** (k=5) avec undersampling si déséquilibre
- Fixer un random seed pour la reproductibilité

**Pourquoi :**
- Garantit des proportions équilibrées dans chaque fold
- Le validation set permet le tuning sans toucher au test set
- La reproductibilité est essentielle pour la recherche

### 9. **VARIABLES PRIORITAIRES IDENTIFIÉES**

**Les plus importantes selon les études :**
- **ESCS** (statut socio-économique et culturel) - systématiquement important
- **HISEI** (statut occupationnel parental) - systématiquement important
- **ICTRES** (accès aux technologies) - très important pour tous domaines
- **METASPAM** (métacognition) - important
- **CPERWEEK** (heures d'enseignement) - important
- **COMPETE** (compétition) - spécifique aux maths

## ⚠️ POINTS CRITIQUES

1. **NE JAMAIS** inclure les PValues dans les features
2. **TOUJOURS** appliquer les transformations (scaling, encoding) APRÈS le split
3. **GÉRER** les >60% de features avec >50% de missing values
4. **PRÉSERVER** l'information ordinale quand elle existe
5. **VALIDER** que les modèles ne sur-apprennent pas avec validation croisée

## 🎯 PIPELINE RECOMMANDÉ

```
1. Analyser le % de missing par variable
2. Éliminer PValues et WLE
3. Éliminer features avec >50% missing
4. Split train/validation/test (60/20/20)
5. Imputation KNN sur train, appliquer sur validation/test
6. Winsorization des outliers
7. Encoding (one-hot pour catégorielles, ordinal pour ordinales)
8. Standardisation/normalisation
9. Feature selection (RFE + MI)
10. Modélisation avec cross-validation
11. Analyse importance (Permutation + SHAP)
```

Cette approche est validée par les 4 études analysées et adaptée à ton contexte spécifique PISA.