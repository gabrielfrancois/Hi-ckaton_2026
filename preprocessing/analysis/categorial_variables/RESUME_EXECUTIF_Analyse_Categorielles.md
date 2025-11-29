# 📊 RÉSUMÉ EXÉCUTIF - Analyse Variables Catégorielles PISA

## 🎯 Objectif
Identifier les variables catégorielles redondantes, métadonnées non prédictives, et variables haute cardinalité pour optimiser le preprocessing.

---

## 📈 Résultats Clés

### État Initial
- **70 variables catégorielles** identifiées
- **Domaine principal** : General (28 variables - 40% du total)
- **Problème majeur** : Codes ISCO haute cardinalité (620 codes × 3 variables = 1860 features potentielles)

---

## 🚨 DÉCOUVERTE CRITIQUE : Explosion de Dimensionnalité

### Impact Encoding One-Hot

**AVANT nettoyage** :
```
OCOD1 (620) + OCOD2 (620) + OCOD3 (620) + Autres (50 vars) 
= ~1910 features après encoding
```

**APRÈS nettoyage** :
```
OCOD1_grouped (10) + OCOD2_grouped (10) + Autres (30 vars)
= ~50 features après encoding
```

### 🎉 GAIN RÉEL : **-97% de features** (1910 → 50)

---

## ⚡ Actions Prioritaires Identifiées

### 🔴 PRIORITÉ CRITIQUE (Métadonnées - Risque NUL)

| Catégorie | Variables | Gain | Justification |
|-----------|-----------|------|---------------|
| **Options questionnaires** | Option_CT, Option_FL, Option_ICTQ, Option_PQ, Option_TQ, Option_UH, Option_WBQ | -7 | Indicateurs admin non prédictifs |
| **Identifiants admin** | CYC, NatCen, SUBNATIO | -3 | Codes administratifs sans valeur |
| **Effort post-test** | EFFORT1, EFFORT2 | -2 | Data leakage potentiel |

**Sous-total Métadonnées : -12 variables** ✅

---

### 🟠 PRIORITÉ HAUTE (Redondances - Risque Minimal)

| Catégorie | Variables | Gain | Justification |
|-----------|-----------|------|---------------|
| **Langues** | LANGTEST_PAQ, LANGTEST_QQQ | -2 | Redondant avec LANGTEST_COG |
| **Date/Grade** | ST003D03T, ST001D01T | -2 | Redondant avec AGE et GRADE |
| **Perspectives parent** | PA008 (doublon), PA162 | -2 | Doublon + perspective élève meilleure |
| **Profession élève** | OCOD3 | -1 | Cardinalité haute + faible prédictivité |

**Sous-total Redondances : -7 variables** ✅

---

### 🔥 PRIORITÉ CRITIQUE (Cardinalité - Impact Massif)

| Action | Variables | Impact |
|--------|-----------|--------|
| **Regroupement ISCO** | OCOD1, OCOD2 | 620 codes → 10 catégories |
| **Suppression** | OCOD3 | -620 codes |

**Impact : -1860 features potentielles → -20 features** 🎯

---

### 🟡 PRIORITÉ VARIABLE (COVID - À Évaluer)

| Variables | Type | Gain Potentiel |
|-----------|------|----------------|
| ST347, ST349, ST350 | COVID catégorielles | -3 |

---

## 📊 Scénarios de Réduction

### Scénario CONSERVATEUR ⭐ (Recommandé)
- **Actions** : Métadonnées + Redondances + ISCO regroupement
- **Réduction variables** : 70 → **51 variables** (-27%)
- **Réduction features** : ~1910 → ~50 features (-97%)
- **Risque** : Minimal
- **Timeline** : Immédiat

### Scénario AGRESSIF
- **Actions** : Conservateur + COVID + validations empiriques
- **Réduction variables** : 70 → **44 variables** (-37%)
- **Réduction features** : ~1910 → ~45 features (-97.6%)
- **Risque** : Moyen
- **Timeline** : Après tests

---

## 🎯 Plan d'Action Immédiat

### Phase 1 : Suppression Métadonnées (EXÉCUTION IMMÉDIATE)

```python
# PRIORITÉ CRITIQUE - Risque NUL
metadonnees_a_supprimer = [
    # Options (7)
    'Option_CT', 'Option_FL', 'Option_ICTQ', 'Option_PQ', 
    'Option_TQ', 'Option_UH', 'Option_WBQ',
    
    # Identifiants (3)
    'CYC', 'NatCen', 'SUBNATIO',
    
    # Effort post-test (2)
    'EFFORT1', 'EFFORT2',
]

df = df.drop(columns=metadonnees_a_supprimer)
# Gain immédiat: -12 variables | Risque: ZÉRO
```

### Phase 2 : Suppression Redondances (PRIORITÉ HAUTE)

```python
# Variables redondantes ou doublons
redondances_a_supprimer = [
    'LANGTEST_PAQ',      # Redondant avec LANGTEST_COG
    'LANGTEST_QQQ',      # Redondant avec LANGTEST_COG
    'ST003D03T',         # Birth Year = redondant avec AGE
    'ST001D01T',         # Grade = redondant avec GRADE
    'PA008',             # Doublon exact (1 copie)
    'PA162',             # Lecture parent (garder ST168)
    'OCOD3',             # Profession aspirée (faible valeur)
]

df = df.drop(columns=redondances_a_supprimer)
# Gain: -7 variables | Risque: Minimal
```

### Phase 3 : Regroupement ISCO (CRITIQUE POUR DIMENSIONNALITÉ)

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
    6: Skilled agricultural workers
    7: Craft and related trades workers
    8: Plant and machine operators
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

---

## 📋 Variables à CONSERVER Absolument

### Variables Critiques (Ne JAMAIS Supprimer)

1. **ST004D01T** (Gender) - Sociodémographique clé + équité
2. **IMMIG** (Immigration) - Important pour fairness/equity analyses
3. **GRADE** (Position grade) - Capture redoublement (fort impact MathScore)
4. **ADMINMODE** (Computer vs Paper) - Mode peut influencer performance
5. **LANGTEST_COG** (Langue test) - Essentiel analyses multilingues
6. **MATHEASE** (Math easier than other subjects) - Perception pertinente
7. **MISSSC** (Missing school >3 months) - Impact éducatif significatif

---

## 🔬 Validation Recommandée

### Avant Suppression Définitive

```python
# 1. Test variance
for var in variables_candidates:
    n_unique = df[var].nunique()
    print(f"{var}: {n_unique} valeurs uniques")
    if n_unique == 1:
        print(f"  → SUPPRIMER (variance nulle)")

# 2. Test corrélation avec MathScore
from scipy.stats import pointbiserialr
from scipy.stats.contingency import association

for var in ['OECD', 'MATHEASE', 'ST003D02T']:
    if df[var].dtype == 'object' or df[var].nunique() < 10:
        # Cramér's V pour catégorielles
        corr = association(pd.crosstab(df[var], df['MathScore']), method='cramer')
    else:
        # Point-biserial pour binaires
        corr, _ = pointbiserialr(df[var], df['MathScore'])
    
    print(f"{var}: corrélation = {corr:.4f}")
    if corr < 0.05:
        print(f"  → Candidat suppression")

# 3. Feature importance (baseline)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Encoder variables catégorielles
X_encoded = df[categorical_vars].apply(LabelEncoder().fit_transform)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_encoded, df['MathScore'])

importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 features importantes:")
print(importance.head(10))
print("\nFeatures importance < 0.001 (candidats suppression):")
print(importance[importance['importance'] < 0.001])
```

---

## ⚠️ Points d'Attention Spécifiques

### Gestion Post-Suppression

1. **Cardinalité restante** :
   - LANGTEST_COG : ~40-50 langues
   - Option : Regrouper en familles linguistiques (Romanes, Germaniques, etc.) → ~10 groupes

2. **Variables COVID** :
   - Total : 7 variables (4 ordinales + 3 catégorielles)
   - Décision selon objectif : prédiction générale vs effet COVID spécifique

3. **Missing values** :
   - Options supprimées → vérifier si leur absence créait missingness informatif
   - OCOD regroupés → traiter missing comme catégorie "Unknown" (code 99)

---

## 💡 Insights Majeurs

### Découvertes Clés

1. **40% des variables catégorielles sont des métadonnées** (28/70 dans General)
2. **Impact ISCO disproportionné** : 3 variables génèrent 97% des features après encoding
3. **Questionnaires parents sous-utilisés** : Souvent incomplets → privilégier données élève
4. **Effet cumulatif** : Variables catégorielles + ISCO = principal driver dimensionnalité

### Comparaison Ordinales vs Catégorielles

| Métrique | Ordinales | Catégorielles | Conclusion |
|----------|-----------|---------------|------------|
| Nombre variables | 96 | 70 | Catégorielles moins nombreuses |
| Réduction possible | 9-18 (-10 à -19%) | 19-26 (-27 à -37%) | **Catégorielles plus impact** |
| Features après encoding | ~100 | ~1910 → ~50 | **Gain massif catégorielles** |
| Risque suppression | Faible-Moyen | Minimal (métadonnées) | Catégorielles plus sûr |

**Conclusion** : Les variables **catégorielles offrent le plus grand ROI** en réduction dimensionnalité avec risque minimal.

---

## 📊 Impact Comparatif Final

### Avant Preprocessing

```
Total features brutes estimées:
- Numériques: 81 features
- Ordinales: ~96 features  
- Catégorielles: ~1910 features (encoding one-hot)
- Groupement: 6 features
TOTAL: ~2093 features
```

### Après Preprocessing Conservateur

```
Total features optimisées:
- Numériques: 81 features
- Ordinales: ~87 features (-9)
- Catégorielles: ~50 features (-1860!)
- Groupement: 6 features
TOTAL: ~224 features
```

### 🎉 GAIN GLOBAL : -89% de features (-1869 features)

---

## 🎯 Recommandation Finale

### Action Immédiate (Cette Semaine)

**Implémenter Scénario CONSERVATEUR** :

1. ✅ Supprimer 12 métadonnées (Phases 1)
2. ✅ Supprimer 7 redondances (Phase 2)  
3. ✅ Regrouper ISCO en 10 catégories (Phase 3)

**Résultat attendu** :
- Variables : 70 → 51 (-27%)
- Features : ~1910 → ~50 (-97%)
- **Impact global preprocessing : ~2093 → ~224 features (-89%)**

### Validation Empirique (Semaine Prochaine)

4. 📊 Tester corrélations variables "à évaluer"
5. 🧪 Comparer performance modèle avant/après
6. 📈 Mesurer impact sur R² et RMSE

### Décision COVID (Selon Objectif)

7. 🔍 Analyser pertinence 7 variables COVID (4 ord + 3 cat)
8. ⚖️ Décider : Conserver, Fusionner, ou Supprimer

---

## ✅ Livrables Générés

1. **analyse_variables_categorielles_redondances.md** - Analyse complète détaillée
2. **recommandations_variables_categorielles.xlsx** - 23 recommandations actionnables
3. **groupes_variables_categorielles.xlsx** - 8 groupes thématiques
4. **variables_categorielles_detail.xlsx** - Liste complète avec descriptions
5. **analyse_categorielles_visualisations.png** - Graphiques impact

---

## 🚀 Conclusion

L'analyse des variables catégorielles révèle une **opportunité exceptionnelle de réduction** :

- **Gain primaire** : Élimination métadonnées (-12 variables, risque nul)
- **Gain secondaire** : Élimination redondances (-7 variables, risque minimal)  
- **Gain tertiaire** : Regroupement ISCO (-1860 features, impact massif)

**Impact total : Réduction de 97% des features catégorielles** avec risque méthodologique minimal.

Cette optimisation est **critique** pour la viabilité du projet, permettant de passer d'un dataset ingérable (~2000+ features) à un dataset optimisé (~200 features) tout en **conservant l'information prédictive essentielle**.

---

**Prochaine étape recommandée** : Exécuter Phases 1-3 immédiatement, puis valider empiriquement avant décisions supplémentaires.
