# Analyse des Variables Ordinales - Redondances et Fusions

## Vue d'ensemble
- **Total de variables ordinales**: 96
- **Objectif**: Identifier les variables redondantes ou similaires pour réduire la dimensionnalité

---

## 🔴 REDONDANCES CRITIQUES À TRAITER

### 1. **Éducation des Parents - DOUBLONS EXACTS**
**Variables concernées**: ST006 (x2) et ST008 (x2)

| Variable | Description | Action |
|----------|-------------|--------|
| ST006 (duplicata 1) | What is the highest ISCED level qualification your mother has obtained? | **SUPPRIMER** |
| ST006 (duplicata 2) | What is the highest ISCED level qualification your mother has obtained? | **CONSERVER** |
| ST008 (duplicata 1) | What is the highest ISCED level of qualification your father has obtained? | **SUPPRIMER** |
| ST008 (duplicata 2) | What is the highest ISCED level of qualification your father has obtained? | **CONSERVER** |

**Décision**: Retirer les doublons (2 variables à supprimer)

---

### 2. **Éducation des Parents - Redondance Fonctionnelle**
**Variables concernées**: ST005, ST006, ST007, ST008

| Variable | Description | Information Captée |
|----------|-------------|-------------------|
| ST005 | What is the highest level of education your mother has completed? | Niveau général mère |
| ST006 | What is the highest ISCED level qualification your mother has obtained? | Niveau ISCED mère (standardisé) |
| ST007 | What is the highest level of schooling completed by your father? | Niveau général père |
| ST008 | What is the highest ISCED level of qualification your father has obtained? | Niveau ISCED père (standardisé) |

**Analyse**: 
- ST005/ST007 = Descriptions générales
- ST006/ST008 = Classification ISCED standardisée (internationale)
- ISCED est plus précis et standardisé pour comparaisons internationales

**Recommandation**: 
- **SUPPRIMER ST005 et ST007** (descriptions générales)
- **CONSERVER ST006 et ST008** (classification ISCED standardisée)
- Gain: -2 variables

---

### 3. **Bien-être et Support Parental - Chevauchement Sémantique**

#### Groupe A: Questions sur le support parental général
| Variable | Description | Domaine |
|----------|-------------|---------|
| PA003 | How often do parents/guardians engage in activities at home that support their child's academic progress, social well-being, and future educational planning? | Support global |
| ST300 | How often do your parents or family members engage in discussions and activities related to your academic life, including your schoolwork, social interactions at school, future education, and general well-being? | Support global |

**Analyse**: Ces deux variables capturent la **même information** (engagement parental global) mais de perspectives différentes (parents vs élève). Fort chevauchement sémantique.

**Recommandation**: 
- **FUSIONNER** en créant un score composite ou **CONSERVER uniquement ST300** (perspective élève plus fiable que questionnaire parents)
- Gain: -1 variable

---

### 4. **Support Parental pendant COVID - Redondance Thématique**

| Variable | Description | Type de Support |
|----------|-------------|----------------|
| ST353 | During COVID closures, how often did family members provide support for your learning in various ways | Support familial COVID |
| ST348 | During COVID closures, how often did your school or teachers engage with you | Support scolaire COVID |
| ST351 | During COVID closures, how often did you use the following learning resources | Ressources COVID |
| ST352 | During COVID closures, how often did you experience challenges | Difficultés COVID |

**Analyse**: 4 variables dédiées exclusivement à COVID. Si COVID n'est pas l'objet principal de l'étude:

**Recommandation**: 
- **Option 1**: Créer un **score COVID composite** (1 seule variable dérivée) → Gain: -3 variables
- **Option 2**: **SUPPRIMER toutes les variables COVID** si non pertinent pour prédire MathScore → Gain: -4 variables

---

### 5. **Ressources Numériques à la Maison - Granularité Excessive**

| Variable | Description | Information |
|----------|-------------|------------|
| ST253 | How many digital devices with screens are present in your home? | Total écrans |
| ST254 | How many of the following digital devices are in your home: televisions, desktop computers, laptop computers or notebooks, tablets, e-book readers, and smartphones? | Détail par type |

**Analyse**: ST254 contient ST253 + détails supplémentaires

**Recommandation**: 
- **SUPPRIMER ST253** (redondant avec ST254)
- Gain: -1 variable

---

### 6. **Livres à la Maison - Double Mesure**

| Variable | Description | Information |
|----------|-------------|------------|
| ST255 | How many books are in your home? | Quantité globale |
| ST256 | What types and how many books do you have at home across the following categories | Détail par catégorie |

**Analyse**: ST256 contient ST255 + typologie

**Recommandation**: 
- **SUPPRIMER ST255** (redondant avec ST256)
- OU **Créer un score composite** si les deux apportent des infos complémentaires
- Gain: -1 variable

---

### 7. **Utilisation des Ressources Numériques - Chevauchement**

#### Utilisation générale vs spécifique
| Variable | Description | Focus |
|----------|-------------|-------|
| IC170 | How often do you use the following at school | Usage à l'école |
| IC171 | How often students use the following out of school | Usage hors école |
| IC173 | How often are digital resources used in lessons across various subjects | Usage par matière |
| IC174 | How often do you use digital resources for learning activities | Usage par activité |
| IC175 | How often do you use digital resources to engage with feedback | Usage feedback |
| IC176 | How often do you use digital resources for various academic activities | Usage académique |
| IC184 | How often do you use digital resources for mathematical tasks | Usage mathématiques |

**Analyse**: 
- **IC170-IC176**: 7 variables sur l'utilisation TIC avec fort chevauchement
- IC184 spécifique mathématiques (pertinent pour MathScore)
- Les autres peuvent contenir beaucoup de redondance

**Recommandation**: 
- **CONSERVER IC184** (spécifique à mathématiques - cible directe)
- **Créer 2-3 scores composites** pour les autres: "Usage_TIC_Ecole", "Usage_TIC_Maison", "Usage_TIC_Apprentissage"
- Gain potentiel: -4 variables

---

### 8. **Lecture - Habitudes Multiples**

| Variable | Description | Perspective |
|----------|-------------|-------------|
| PA160 | How often do you (parent) choose to read | Habitudes parent |
| ST167 | How often do you (student) read for pleasure | Habitudes élève |

**Analyse**: Questions similaires mais perspectives différentes (peut être complémentaire pour effet modélisation familiale)

**Recommandation**: 
- **CONSERVER les deux** SI on veut modéliser l'influence parentale
- **SUPPRIMER PA160** SI seules les habitudes de l'élève comptent
- Gain potentiel: -1 variable

---

### 9. **Pratiques Pédagogiques - Redondance Thématique**

| Variable | Description | Focus |
|----------|-------------|-------|
| ST100 | Does the teacher demonstrate support for student learning | Support enseignant |
| ST270 | How often does the teacher actively support and ensure student understanding | Support enseignant |

**Analyse**: Deux variables mesurant le **même construit** (support enseignant)

**Recommandation**: 
- **FUSIONNER** en score composite OU **SUPPRIMER ST270** (redondant)
- Gain: -1 variable

---

### 10. **Perturbations en Classe - Duplication**

| Variable | Description | Focus |
|----------|-------------|-------|
| ST097 | Issues with student behavior during test language lessons | Perturbations cours langue |
| ST273 | Classroom disruptions impede effective learning | Perturbations générales |

**Analyse**: ST273 = version plus générale de ST097

**Recommandation**: 
- **CONSERVER ST273** (plus général, applicable à toutes matières)
- **SUPPRIMER ST097** (spécifique cours langue, moins pertinent pour MathScore)
- Gain: -1 variable

---

## 📊 ANALYSE PAR DOMAINE THÉMATIQUE

### Domaine: Family Background & Socioeconomic Status (22 → 15 variables)

#### Variables à SUPPRIMER (7)
1. **ST005** - Redondant avec ST006 (ISCED mère)
2. **ST006 (duplicata)** - Doublon exact
3. **ST007** - Redondant avec ST008 (ISCED père)
4. **ST008 (duplicata)** - Doublon exact
5. **ST253** - Redondant avec ST254 (écrans)
6. **ST255** - Redondant avec ST256 (livres)
7. **ST300** - Fusionner avec PA003 ou supprimer

**Gain: -7 variables**

---

### Domaine: Classroom Environment & Teaching Practices (13 → 10 variables)

#### Variables à SUPPRIMER/FUSIONNER (3)
1. **ST097** - Supprimer (redondant avec ST273)
2. **ST100 + ST270** - Fusionner en score composite (support enseignant)

**Gain: -3 variables**

---

### Domaine: ICT Use & Digital Competence (11 → 7 variables)

#### Stratégie de consolidation
- **CONSERVER**: IC184 (usage mathématiques - pertinent direct)
- **CRÉER 3 SCORES COMPOSITES**:
  1. Score_TIC_Infrastructure (IC172, IC170)
  2. Score_TIC_Activités (IC173, IC174, IC175, IC176)
  3. Score_Dépendance_Numérique (ST322)

**Gain: -4 variables**

---

### Domaine: Variables COVID (4 → 0-1 variables)

#### Option A (Conservatrice)
- Créer **1 score composite COVID**
- Gain: -3 variables

#### Option B (Agressive)
- **SUPPRIMER toutes** si COVID pas pertinent pour prédire MathScore
- Gain: -4 variables

---

## 🎯 SYNTHÈSE DES RECOMMANDATIONS

### Scénario CONSERVATEUR (Réduction modérée)
| Action | Nombre Variables |
|--------|-----------------|
| Suppression doublons exacts | -2 |
| Suppression éducation parents (ST005, ST007) | -2 |
| Suppression ressources numériques (ST253, ST255) | -2 |
| Suppression perturbations (ST097) | -1 |
| Fusion support parental (PA003/ST300) | -1 |
| Fusion support enseignant (ST100/ST270) | -1 |
| **TOTAL RÉDUCTION** | **-9 variables** |
| **TOTAL FINAL** | **87 variables ordinales** |

---

### Scénario AGRESSIF (Réduction maximale)
| Action | Nombre Variables |
|--------|-----------------|
| Toutes actions scénario conservateur | -9 |
| Suppression variables COVID | -4 |
| Consolidation TIC (scores composites) | -4 |
| Suppression habitudes lecture parent (PA160) | -1 |
| **TOTAL RÉDUCTION** | **-18 variables** |
| **TOTAL FINAL** | **78 variables ordinales** |

---

## 📋 PLAN D'ACTION RECOMMANDÉ

### Phase 1: Nettoyage Critique (Priorité HAUTE)
```python
variables_a_supprimer_phase1 = [
    'ST006_duplicata',  # Doublon exact
    'ST008_duplicata',  # Doublon exact
    'ST005',           # Redondant avec ST006
    'ST007',           # Redondant avec ST008
    'ST253',           # Redondant avec ST254
    'ST255',           # Redondant avec ST256
    'ST097',           # Redondant avec ST273
]
```
**Impact**: -7 variables / Risque: Minimal

---

### Phase 2: Consolidation (Priorité MOYENNE)
```python
fusions_a_creer = {
    'Support_Parental': ['PA003', 'ST300'],  # Moyenne ou PCA
    'Support_Enseignant': ['ST100', 'ST270'],  # Moyenne ou PCA
}
```
**Impact**: -2 variables / Risque: Faible

---

### Phase 3: Réévaluation COVID (Priorité VARIABLE)
```python
variables_covid = ['ST348', 'ST351', 'ST352', 'ST353']

# Option 1: Score composite
# Option 2: Suppression complète si non pertinent
```
**Impact**: -3 ou -4 variables / Risque: Dépend du contexte

---

### Phase 4: Optimisation TIC (Priorité BASSE - après tests)
```python
consolidation_tic = {
    'Score_TIC_Infra': ['IC170', 'IC172'],
    'Score_TIC_Usage': ['IC173', 'IC174', 'IC175', 'IC176'],
    # CONSERVER IC184 séparément
}
```
**Impact**: -4 variables / Risque: Moyen (perte d'information granulaire)

---

## ⚠️ POINTS D'ATTENTION

1. **Validation empirique**: Tester la corrélation entre variables avant suppression
2. **Importance pour MathScore**: Vérifier l'importance de chaque variable dans des modèles préliminaires
3. **Informations culturelles**: Certaines variables (langues, livres) peuvent capturer des nuances culturelles importantes
4. **Questionnaires parents vs élèves**: Les perspectives peuvent être complémentaires même si les questions se ressemblent

---

## 📈 GAIN ESTIMÉ

| Scénario | Variables Initiales | Variables Finales | Réduction |
|----------|-------------------|------------------|-----------|
| **Conservateur** | 96 | 87 | -9.4% |
| **Agressif** | 96 | 78 | -18.8% |

**Recommandation finale**: Commencer par le **scénario conservateur** (Phase 1-2), puis évaluer l'impact avant de procéder aux phases 3-4.