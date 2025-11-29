**Source**: Glossaire PISA (Programme for International Student Assessment)  
**Nombre total de features**: 308 variables principales mais 3 dupliqués donc 305 variables  
**Structure**: 6 feuilles (General, CNT, CNTRYID, STRATUM, ISCEDP, OCOD)  

---

### 1. **STRUCTURE ET MÉTADONNÉES**

- **16 domaines thématiques** différents  
- **6 feuilles de référence** pour les variables catégorielles encodées  
- La **cohérence des noms de colonnes**  
- Les **patterns de nommage** (préfixes ST, PA, IC, WB, etc.)  
- La **longueur des codes** (3-25 caractères, médiane à 5)  

---

### 2. **TYPES DE VARIABLES**

#### Distribution par domaine thématique :
```
Math                                           43 variables (14%)
General                                        38 variables (12%)
Science                                        38 variables (12%)
Family Background & Socioeconomic Status       31 variables (10%)
Reading                                        30 variables (10%)
```

#### Patterns de codes identifiés :
- **ST** (94 variables) : Questions sur les étudiants (Student)
- **math/science/reading** (110 variables) : Domaines de compétence
- **PA** (30 variables) : Questions parentales (Parent)
- **IC** (14 variables) : Compétences TIC (ICT)
- **WB** (13 variables) : Bien-être (Well-Being)

---

### 3. **QUALITÉ DES DONNÉES**

#### ✅ Problèmes détectés :
- **1 valeur manquante** dans la colonne `description`
- **6 codes dupliqués** (ST006, ST008, PA008) - nécessite investigation
- **Variables de référence** :
  - 80 pays (CNT/CNTRYID)
  - 1316 strates différentes (STRATUM)
  - 59 niveaux ISCED (classification éducative)
  - 620 codes de professions (OCOD)

---

### 4. **VARIABLES CLÉS POUR LE PREPROCESSING**

#### A. **Identifiants et variables structurelles** (CRITIQUES)
```
Year           : Année du test
CNT            : Code pays (3 caractères)
CNTRYID        : Identifiant numérique du pays
CNTSCHID       : ID international de l'école
CNTSTUID       : ID international de l'étudiant
STRATUM        : Strate d'échantillonnage
```
➡️ **Action** : Ne jamais supprimer, vérifier unicité des ID

#### B. **Variables démographiques**
```
ST004         : Genre (probablement)
ST006/ST008   : Niveau éducatif parental (dupliqué - à investiguer)
ST230         : Nombre de frères/sœurs
ST253         : Nombre d'appareils numériques
ST255         : Nombre de livres à la maison
```
➡️ **Action** : Imputation stratifiée si manquantes, création de catégories

#### C. **Variables de performance** (43 en Math, 38 en Science, 30 en Reading)
➡️ **Action** : 
- Vérifier les échelles de notation
- Détecter les valeurs aberrantes
- Possibles valeurs plafond/plancher

#### D. **Variables psychométriques**
```
Motivation, Mindset & Self-Regulation     : 13 variables
Student Well-Being & Mental Health        : 8 variables
Social-Emotional Competencies             : 6 variables
```
➡️ **Action** : Souvent des échelles de Likert - vérifier cohérence interne

---

### 5. **RELATIONS ET HIÉRARCHIES**

#### Hiérarchie géographique identifiée :
```
CNTRYID → CNT → STRATUM
(Pays → Code → Stratification urbain/rural, public/privé, régionale)
```

#### Hiérarchie éducative (ISCEDP) :
```
010 : Early childhood
100 : Primary
241-244 : Lower secondary (General)
251-254 : Lower secondary (Vocational)
...
```

---

### 6. **DICTIONNAIRE DE CODAGE**

#### Feuilles de référence cruciales :
1. **CNT** : Mapping codes pays ↔ noms (80 pays)
2. **CNTRYID** : Mapping ID numériques ↔ noms pays
3. **STRATUM** : Description de 1316 strates (ex: "ALB01: Urban/North/Public")
4. **ISCEDP** : 59 niveaux éducatifs standardisés UNESCO
5. **OCOD** : 620 codes de professions (classification ISCO)

---

### 7. **Catégories de features**

| Type de Variable | Nombre | Pourcentage |  |
|-----------------|--------|-------------|------------|
| **Variables Numériques** | 135 | 43.8% | Valeurs quantitatives continues ou discrètes mesurables |
| **Variables Ordinales** | 96 | 31.2% | Catégories avec un ordre logique (échelles, fréquences) |
| **Variables Catégorielles** | 70 | 22.7% | Catégories sans ordre (modalités nominales) |
| **Variables de Groupement** | 7 | 2.3% | Identifiants pour structurer et agréger les données |
