# Analyse des Variables Catégorielles - Redondances et Fusions

## Vue d'ensemble
- **Total de variables catégorielles**: 70
- **Objectif**: Identifier les variables redondantes, métadonnées non prédictives, ou fusionnables

---

## 🔴 REDONDANCES CRITIQUES À TRAITER

### 1. **Options Questionnaires - Variables Indicatrices (7 variables)**

| Variable | Description | Utilité Prédictive |
|----------|-------------|-------------------|
| Option_CT | Creative Thinking Option (Yes/No) | **MÉTADONNÉE** |
| Option_FL | Financial Literacy Option (Yes/No) | **MÉTADONNÉE** |
| Option_ICTQ | ICT Questionnaire Option (Yes/No) | **MÉTADONNÉE** |
| Option_PQ | Parent Questionnaire Option (Yes/No) | **MÉTADONNÉE** |
| Option_TQ | Teacher Questionnaire Option (Yes/No) | **MÉTADONNÉE** |
| Option_UH | Une Heure Option (Yes/No) | **MÉTADONNÉE** |
| Option_WBQ | Well-Being Questionnaire Option (Yes/No) | **MÉTADONNÉE** |

**Analyse**: Ces variables indiquent simplement **quels questionnaires ont été administrés** dans chaque pays/école. Ce sont des métadonnées administratives, PAS des caractéristiques des élèves.

**Impact sur MathScore**: NUL - Ces variables n'ont aucun lien causal avec la performance mathématique.

**Recommandation**: 
- **SUPPRIMER TOUTES** (7 variables)
- **Alternative conservatrice**: Les conserver uniquement pour analyse de missingness (si questionnaire non administré = données manquantes)
- Gain: **-7 variables** ✅

---

### 2. **Identifiants/Codes Administratifs (4 variables)**

| Variable | Description | Type | Recommandation |
|----------|-------------|------|----------------|
| CYC | PISA Assessment Cycle | Identifiant temporel | **SUPPRIMER** |
| NatCen | National Centre 6-digit Code | Identifiant administratif | **SUPPRIMER** |
| OECD | OECD country (Yes/No) | Indicateur géographique | **ÉVALUER** |
| SUBNATIO | Sub-region code (7 digits) | Identifiant géographique | **REDONDANT** |

**Analyse détaillée**:
- **CYC**: Tous les élèves du dataset sont du même cycle → variance = 0 → inutile
- **NatCen**: Code administratif sans signification prédictive
- **OECD**: Peut capturer différences économiques/systèmes éducatifs (À TESTER)
- **SUBNATIO**: Déjà capturé par CNT (pays) + STRATUM → redondant

**Recommandation**: 
- **SUPPRIMER**: CYC, NatCen, SUBNATIO (3 variables)
- **CONSERVER**: OECD si corrélation significative avec MathScore
- Gain: **-3 variables** minimum

---

### 3. **Langues du Test - Redondance Partielle (3 variables)**

| Variable | Description | Information |
|----------|-------------|------------|
| LANGTEST_COG | Language of Assessment (test cognitif) | Langue principale |
| LANGTEST_PAQ | Language of Parent Questionnaire | Langue questionnaire parent |
| LANGTEST_QQQ | Language of Questionnaire (élève) | Langue questionnaire élève |

**Analyse**: 
- Dans la majorité des cas: LANGTEST_COG = LANGTEST_QQQ = langue du pays
- LANGTEST_PAQ peut différer (familles immigrées)
- Information principale = concordance langue maison/test

**Recommandation**: 
- **CRÉER variable dérivée**: "Language_Mismatch" (Oui/Non - langue test ≠ langue maison)
- **SUPPRIMER**: LANGTEST_PAQ, LANGTEST_QQQ
- **CONSERVER**: LANGTEST_COG (langue du test)
- Gain: **-2 variables** (ou -1 si on crée Language_Mismatch)

---

### 4. **Professions Parents (ISCO) - Haute Cardinalité (3 variables)**

| Variable | Description | Cardinalité | Problème |
|----------|-------------|-------------|----------|
| OCOD1 | ISCO-08 Occupation Mother | ~620 codes | **TRÈS HAUTE** |
| OCOD2 | ISCO-08 Occupation Father | ~620 codes | **TRÈS HAUTE** |
| OCOD3 | ISCO-08 Occupation Self (aspiration élève) | ~620 codes | **TRÈS HAUTE** |

**Analyse**: 
- 620 codes professionnels ISCO-08 = cardinalité explosive
- Encodage one-hot impossible (620 colonnes par variable!)
- Information déjà partiellement capturée par éducation parents (ST006, ST008)

**Options**:

**Option A - Regroupement ISCO**:
```python
# Utiliser les 2 premiers chiffres ISCO (10 grandes catégories)
OCOD1_grouped = OCOD1 // 10  # Managers, Professionals, Technicians...
# Gain: Cardinalité 620 → 10
```

**Option B - Score socio-économique**:
```python
# Créer score SES composite avec OCOD + Education + Ressources
SES_Score = f(OCOD1, OCOD2, ST006, ST008, ST251, ST255)
# Supprimer OCOD1, OCOD2 individuellement
```

**Option C - Suppression pure**:
- OCOD3 (aspiration élève) → faible lien avec performance actuelle
- Conserver OCOD1, OCOD2 regroupés

**Recommandation**: 
- **REGROUPER OCOD1 et OCOD2** en 10 catégories ISCO principales
- **SUPPRIMER OCOD3** (aspiration future, non prédictif performance actuelle)
- Gain effectif: **-1 variable** + réduction massive cardinalité (620 → 10)

---

### 5. **Date de Naissance - Granularité Excessive (2 variables)**

| Variable | Description | Utilité |
|----------|-------------|---------|
| ST003D02T | Birth Month | Granularité excessive |
| ST003D03T | Birth Year | Redondant avec AGE |

**Analyse**: 
- **Birth Year** → directement capturé par variable AGE (numérique)
- **Birth Month** → peut capturer "relative age effect" (mois dans l'année scolaire)
  - Mais très faible effet documenté dans littérature PISA
  - Spécifique aux systèmes avec dates de coupure strictes

**Recommandation**: 
- **SUPPRIMER ST003D03T** (Birth Year - redondant avec AGE)
- **ÉVALUER ST003D02T** (Birth Month - tester corrélation)
  - Si corrélation faible → SUPPRIMER
  - Si significatif → CONSERVER
- Gain: **-1 à -2 variables**

---

### 6. **Grade/Niveau Scolaire - Redondance (2 variables)**

| Variable | Description | Information |
|----------|-------------|------------|
| GRADE | Grade compared to modal grade | Position relative |
| ST001D01T | Student International Grade (Derived) | Grade absolu |

**Analyse**: 
- ST001D01T = grade réel de l'élève
- GRADE = écart par rapport au grade modal du pays
- GRADE capture "être en avance/retard" (redoublement/saut de classe)
- Information de GRADE peut être dérivée: GRADE = ST001D01T - modal_grade(CNT)

**Recommandation**: 
- **CONSERVER GRADE** (capture effet redoublement/avance - important pour MathScore)
- **SUPPRIMER ST001D01T** (redondant - dérivable de GRADE + CNT)
- Gain: **-1 variable**

---

### 7. **Effort sur le Test PISA (2 variables)**

| Variable | Description | Problème |
|----------|-------------|----------|
| EFFORT1 | Effort put into this test | Auto-déclaré **APRÈS** le test |
| EFFORT2 | Effort if results counted for grades | Hypothétique |

**Analyse**: 
- Ces variables mesurent **l'effort auto-déclaré** APRÈS le test
- **Biais circulaire potentiel**: 
  - Élèves qui ont bien réussi → déclarent plus d'effort
  - Élèves qui ont mal réussi → sous-estiment leur effort
- EFFORT2 = question hypothétique ("et si ça comptait pour la note?")

**Problème éthique/méthodologique**:
- Utiliser EFFORT comme prédicteur de MathScore = **contamination**
- L'effort déclaré est influencé par la perception de réussite

**Recommandation**: 
- **SUPPRIMER EFFORT1 et EFFORT2** pour éviter data leakage
- Ces variables sont post-hoc, non des caractéristiques pré-existantes
- Gain: **-2 variables** ✅

---

### 8. **Doublon Exact Détecté**

| Variable | Description | Problème |
|----------|-------------|----------|
| PA008 (ligne 1) | Parent involvement with school | **DOUBLON** |
| PA008 (ligne 2) | Parent involvement with school | **DOUBLON** |

**Recommandation**: 
- **SUPPRIMER 1 des 2 doublons PA008**
- Gain: **-1 variable** ✅

---

### 9. **Approche Lecture - Même Question (2 variables)**

| Variable | Description | Perspective |
|----------|-------------|-------------|
| PA162 | Parent: typical approach to reading books | Parent |
| ST168 | Student: typical approach to reading books | Élève |

**Analyse**: 
- **Même question exactement**, 2 perspectives différentes
- Similaire à PA160/ST167 (habitudes lecture) dans les ordinales
- Perspective parent souvent moins fiable (questionnaire partiellement rempli)

**Recommandation**: 
- **SUPPRIMER PA162** (perspective parent)
- **CONSERVER ST168** (perspective élève plus fiable)
- Gain: **-1 variable**

---

### 10. **Variables Spécifiques COVID (2 variables)**

| Variable | Description | Domaine |
|----------|-------------|---------|
| ST347 | School closures in last 3 years (COVID or other) | COVID |
| ST349 | Main digital device during COVID | COVID |
| ST350 | Amount of learning during COVID vs normal | COVID |

**Analyse**: 
- 3 variables catégorielles COVID (en plus des 4 ordinales identifiées précédemment)
- Total COVID: **7 variables** (4 ordinales + 3 catégorielles)
- Pertinence dépend de l'objectif: prédire MathScore en général ou effet COVID?

**Recommandation**: 
- **Si COVID non pertinent**: SUPPRIMER les 3 catégorielles
- **Si COVID pertinent**: Créer 1-2 scores composites COVID globaux
- Gain potentiel: **-2 à -3 variables**

---

## 📊 ANALYSE PAR DOMAINE THÉMATIQUE

### Domaine: General (28 → 13 variables) ⚠️ RÉDUCTION MAJEURE

#### Variables à SUPPRIMER (15)
1. **Options questionnaires** (7): Option_CT, Option_FL, Option_ICTQ, Option_PQ, Option_TQ, Option_UH, Option_WBQ
2. **Identifiants administratifs** (3): CYC, NatCen, SUBNATIO
3. **Langues** (2): LANGTEST_PAQ, LANGTEST_QQQ
4. **Métadonnées date** (1): ST003D03T (Birth Year)
5. **Grade** (1): ST001D01T (redondant avec GRADE)
6. **Effort** (2): EFFORT1, EFFORT2

#### Variables à REGROUPER
- **OCOD1, OCOD2**: Regrouper en 10 catégories ISCO (au lieu de 620)
- **OCOD3**: SUPPRIMER

**Gain domaine General: -15 variables + réduction cardinalité massive**

---

### Domaine: Career Exploration & Future Orientation (9 → 8 variables)

**Analyse**: Peu de redondances détectées dans ce domaine.

**Recommandation conservatrice**: 
- Toutes les variables semblent apporter information distincte
- **Aucune suppression immédiate recommandée**

**Recommandation agressive**:
- **PA032** et **PA197** (perspectives parents sur carrières enfant) → peut-être moins prédictif que perspectives élève
- Gain potentiel: -1 à -2 variables

---

### Domaine: Educational History & Trajectory (7 → 5 variables)

#### Variables COVID à évaluer
- **ST347** (School closures - COVID)
- **ST350** (Learning during COVID)

**Recommandation**: 
- **SUPPRIMER ST347 et ST350** si COVID non pertinent
- Gain: **-2 variables**

---

### Domaine: Family Background & Socioeconomic Status (7 → 6 variables)

#### Doublon détecté
- **PA008** (doublon exact)

**Recommandation**: 
- **SUPPRIMER 1 doublon PA008**
- Gain: **-1 variable**

---

### Domaine: Reading Engagement & Literacy Practices (4 → 3 variables)

#### Perspectives parent/élève
- **PA162 vs ST168** (approche lecture)

**Recommandation**: 
- **SUPPRIMER PA162** (perspective parent moins fiable)
- Gain: **-1 variable**

---

### Domaine: ICT Use & Digital Competence (4 → 3 variables)

#### Variable COVID
- **ST349** (Main device during COVID)

**Recommandation**: 
- **SUPPRIMER ST349** si COVID non pertinent
- Gain: **-1 variable**

---

### Domaines Restants (11 variables)

**Motivation, Mindset & Self-Regulation** (3 variables): Pas de redondance évidente
**Student Well-Being & Mental Health** (2 variables): Variables distinctes
**Social-Emotional Competencies** (2 variables): Variables distinctes
**Classroom Environment** (2 variables): Variables distinctes
**Mathematics Learning** (2 variables): Variables distinctes

**Recommandation**: CONSERVER toutes (information unique)

---

## 🎯 SYNTHÈSE DES RECOMMANDATIONS

### Scénario CONSERVATEUR (Réduction Sûre)

| Catégorie | Variables | Gain |
|-----------|-----------|------|
| Options questionnaires | Option_* (7 variables) | -7 |
| Identifiants admin | CYC, NatCen, SUBNATIO | -3 |
| Langues | LANGTEST_PAQ, LANGTEST_QQQ | -2 |
| Date naissance | ST003D03T | -1 |
| Grade | ST001D01T | -1 |
| Effort test | EFFORT1, EFFORT2 | -2 |
| Profession élève | OCOD3 | -1 |
| Doublon | PA008 (1 copie) | -1 |
| Lecture | PA162 | -1 |
| **TOTAL CONSERVATEUR** | | **-19 variables** |
| **RÉSULTAT FINAL** | | **51 variables** |

**Réduction: 70 → 51 variables (-27%)**

---

### Scénario AGRESSIF (Réduction Maximale)

| Ajouts au scénario conservateur | Variables | Gain |
|----------------------------------|-----------|------|
| Variables COVID catégorielles | ST347, ST349, ST350 | -3 |
| Birth Month (faible effet) | ST003D02T | -1 |
| OECD (si non significatif) | OECD | -1 |
| Carrières perspectives parents | PA032, PA197 | -2 |
| **TOTAL AGRESSIF** | | **-26 variables** |
| **RÉSULTAT FINAL** | | **44 variables** |

**Réduction: 70 → 44 variables (-37%)**

---

## 📋 PLAN D'ACTION RECOMMANDÉ

### Phase 1: Nettoyage Métadonnées (PRIORITÉ CRITIQUE) ✅

```python
# Variables à supprimer IMMÉDIATEMENT (métadonnées non prédictives)
metadonnees_a_supprimer = [
    # Options questionnaires (7)
    'Option_CT', 'Option_FL', 'Option_ICTQ', 'Option_PQ', 
    'Option_TQ', 'Option_UH', 'Option_WBQ',
    
    # Identifiants administratifs (3)
    'CYC', 'NatCen', 'SUBNATIO',
    
    # Effort post-test (2)
    'EFFORT1', 'EFFORT2',
]
# Gain: -12 variables | Risque: ZÉRO
```

### Phase 2: Redondances Fonctionnelles (PRIORITÉ HAUTE) ✅

```python
redondances_a_supprimer = [
    'LANGTEST_PAQ',     # Redondant avec LANGTEST_COG
    'LANGTEST_QQQ',     # Redondant avec LANGTEST_COG
    'ST003D03T',        # Birth Year - redondant avec AGE
    'ST001D01T',        # Grade absolu - redondant avec GRADE
    'PA008',            # Doublon exact (garder 1 seule copie)
    'PA162',            # Approche lecture parent (garder ST168)
    'OCOD3',            # Profession aspirée élève (faible prédictivité)
]
# Gain: -7 variables | Risque: Minimal
```

### Phase 3: Regroupement Cardinalité (PRIORITÉ HAUTE) ⚙️

```python
# Regrouper codes ISCO en 10 catégories principales
def group_isco_codes(ocod):
    """
    Regrouper 620 codes ISCO-08 en 10 grandes catégories:
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
    return ocod // 100  # Utiliser 1er chiffre ISCO
    
OCOD1_grouped = group_isco_codes(OCOD1)
OCOD2_grouped = group_isco_codes(OCOD2)

# Impact: Cardinalité 620 → 10 par variable
# Gain effectif: Réduction explosive du nombre de features après encoding
```

### Phase 4: Évaluation COVID (PRIORITÉ VARIABLE)

```python
variables_covid_categorielles = ['ST347', 'ST349', 'ST350']

# Option 1: SUPPRIMER si COVID non pertinent
# Option 2: FUSIONNER avec variables COVID ordinales en scores composites

# Décision basée sur:
# - Objectif de l'étude (prédiction générale vs effet COVID)
# - Corrélation avec MathScore
# - Distribution temporelle du dataset
```

---

## 🔬 VALIDATION RECOMMANDÉE

### Tests à effectuer AVANT suppression définitive

1. **Test de variance**:
```python
# Vérifier variance des variables identifiées
for var in ['CYC', 'OECD', 'ST003D02T']:
    print(f"{var}: {df[var].nunique()} valeurs uniques")
    # Si nunique = 1 → suppression immédiate
```

2. **Test de corrélation avec MathScore**:
```python
# Pour variables "à évaluer"
for var in ['OECD', 'ST003D02T', 'MATHEASE']:
    correlation = point_biserial_or_cramers_v(df[var], df['MathScore'])
    print(f"{var}: corrélation = {correlation}")
    # Si corrélation < 0.05 → supprimer
```

3. **Test d'importance dans modèle baseline**:
```python
# Random Forest rapide pour feature importance
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_encoded, y)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Variables avec importance < 0.001 → candidats suppression
```

---

## ⚠️ POINTS D'ATTENTION CRITIQUES

### Variables à NE JAMAIS SUPPRIMER

1. **ST004D01T** (Gender) - Variable sociodémographique clé
2. **IMMIG** (Immigration status) - Important pour équité/fairness
3. **GRADE** (Position grade) - Capture effet redoublement (fort impact)
4. **ADMINMODE** (Computer vs Paper) - Peut influencer performance
5. **LANGTEST_COG** (Langue du test) - Important pour analyses multilingues

### Gestion Cardinalité Post-Suppression

Après suppressions, cardinalité restante:
- **OCOD1, OCOD2**: 620 → 10 (après regroupement) ✅
- **LANGTEST_COG**: ~40-50 langues → CONSERVER ou regrouper en familles linguistiques
- **Autres variables catégorielles**: Cardinalité généralement < 10

---

## 📈 IMPACT ATTENDU

### Réduction Nombre de Features Après Encoding

**One-Hot Encoding - Avant nettoyage**:
```
OCOD1 (620) + OCOD2 (620) + OCOD3 (620) + autres (50) 
= ~1900 colonnes après encoding
```

**One-Hot Encoding - Après nettoyage conservateur**:
```
OCOD1_grouped (10) + OCOD2_grouped (10) + autres (30)
= ~50 colonnes après encoding
```

**Gain réel: ~1850 colonnes en moins!** 🎉

---

## 💡 INSIGHTS CLÉS

### Découvertes Majeures

1. **Métadonnées massives** : 27% des variables catégorielles (19/70) sont des métadonnées non prédictives
2. **Explosion ISCO** : 3 variables OCOD génèrent 1860 features après encoding → réduction à 30 features
3. **COVID surreprésenté** : 7 variables totales (4 ordinales + 3 catégorielles) sur contexte temporaire
4. **Perspectives parent/élève** : Doublons systématiques → privilégier perspective élève

### Recommandations Méthodologiques

1. **Toujours regrouper codes haute cardinalité** (ISCO, professions, codes géo)
2. **Supprimer métadonnées administratives** (options, identifiants, cycles)
3. **Éliminer variables post-hoc** (EFFORT mesuré après le test)
4. **Privilégier perspective élève** sur perspective parent (plus fiable)

---

## 🎯 SYNTHÈSE FINALE

### Recommandation Principale

**Implémenter Scénario CONSERVATEUR (Phases 1-2)**:
- Suppression: 19 variables
- Regroupement ISCO: 620 → 10 codes
- **Résultat: 70 → 51 variables catégorielles**
- **Impact réel après encoding: ~1900 → ~50 features**
- **Réduction totale: ~97% de features en moins!**

### Validation Empirique Nécessaire

Avant Phase 3-4 (COVID, variables additionnelles):
1. Tester corrélations avec MathScore
2. Mesurer feature importance
3. Comparer performance modèles avec/sans variables candidates

---

**Conclusion**: Le nettoyage des variables catégorielles offre le **plus grand gain** en termes de réduction de dimensionnalité, principalement grâce au regroupement des codes ISCO haute cardinalité et à l'élimination des métadonnées.
