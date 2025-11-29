# Tutoriel : Lancer le Preprocessing Complet

Ce tutoriel explique comment lancer l'ensemble du pipeline de preprocessing sur les données X_train.

## Vue d'ensemble du pipeline

Le preprocessing se déroule en 2 étapes principales :
1. **Étape 1** : Preprocessing des variables numériques et de groupement (par Gabriel)
2. **Étape 2** : Preprocessing des variables ordinales et catégorielles (par Alizée)

---

## Prérequis

### 1. Placer X_train au bon endroit

Le fichier `X_train.csv` doit être placé dans le dossier `data/` à la racine du projet :

```
Code_Hickathon/
├── data/
│   └── X_train.csv          ← Placer le fichier ici
├── preprocessing/
│   └── ...
```

**Vérification** : Le script [load.py](preprocessing/preprocessing_numerical_grouped_variables/load.py#L168) s'attend à trouver le fichier à `../../data/X_train.csv` (relatif au dossier `preprocessing/preprocessing_numerical_grouped_variables/`).

---

## Étape 1 : Preprocessing des variables numériques et de groupement

### Script à lancer

```bash
python preprocessing/preprocessing_numerical_grouped_variables/concat.py
```

### Ce que fait ce script

1. **Charge** les données depuis `data/X_train.csv`
2. **Sélectionne** uniquement les colonnes numériques et de groupement (définies dans [load.py](preprocessing/preprocessing_numerical_grouped_variables/load.py))
   - Variables numériques : 135 colonnes (AGE, scores de lecture/math/science, timings, etc.)
   - Variables de groupement : 6 colonnes (Year, CNT, CNTRYID, CNTSCHID, CNTSTUID, STRATUM)
3. **Préprocesse** ces colonnes via les modules `numerical.py` et `grouped.py`
4. **Concatène** les colonnes preprocessées avec les autres colonnes non modifiées (ordinales et catégorielles)
5. **Sauvegarde** le résultat dans `data/X_numerical_grouped_cleaned_train.csv`

### Sortie attendue

Fichier créé : `data/X_numerical_grouped_cleaned_train.csv`

Ce fichier contient :
- Toutes les colonnes de X_train
- Les colonnes numériques et de groupement ont été preprocessées
- Les colonnes ordinales et catégorielles sont inchangées

**Dimensions** : Même nombre de lignes que X_train (1 172 086), nombre de colonnes peut varier selon le preprocessing appliqué.

---

## Étape 2 : Preprocessing des variables ordinales et catégorielles

### Script à lancer

```bash
python preprocessing/tests/run_preprocessing_on_data.py
```

### Ce que fait ce script

1. **Charge** `data/X_numerical_grouped_cleaned_train.csv` (sortie de l'étape 1)
2. **Identifie** les variables ordinales et catégorielles présentes
   - Variables ordinales : ST005, ST006, ST007, ST008, ST097, ST100, ST253, ST254, ST255, ST256, ST270, ST273, PA003, ST300
   - Variables catégorielles : Option_CT, Option_FL, Option_ICTQ, Option_PQ, Option_TQ, Option_UH, Option_WBQ, CYC, NatCen, SUBNATIO, LANGTEST_PAQ, LANGTEST_COG, LANGTEST_QQQ, ST003D03T, ST001D01T, OCOD1, OCOD2, OCOD3, PA008, PA162
3. **Preprocessing ordinal** :
   - Création de scores composites (moyennes de variables liées)
   - Suppression de variables redondantes
   - Imputation par la médiane
4. **Preprocessing catégoriel** :
   - Suppression des métadonnées
   - Suppression des redondances
   - Regroupement des codes ISCO (professions)
   - Imputation par le mode ou "Unknown"
5. **Suppression** des variables avec plus de 50% de valeurs manquantes
6. **Imputation générique** de toutes les valeurs manquantes restantes :
   - Colonnes numériques : médiane
   - Colonnes catégorielles : mode ou "Unknown"
7. **Sauvegarde** dans `data/X_train_preprocessed_{date}_{heure}.csv`

### Sortie attendue

Fichier créé : `data/X_train_preprocessed_YYYYMMDD_HHMMSS.csv`

Exemple : `data/X_train_preprocessed_20251129_180836.csv`

Ce fichier contient :
- **Toutes les variables** ont été preprocessées
- **Aucune valeur manquante** (toutes imputées)
- **Colonnes réduites** : suppression des variables avec >50% missing et des redondances

Le script affiche également un rapport détaillé avec :
- Dimensions finales
- Nombre de colonnes supprimées
- Variables ordinales transformées
- Variables catégorielles transformées
- Statistiques d'imputation

---

## Pipeline complet en 2 commandes

```bash
# Étape 1 : Preprocessing numériques + groupement
python preprocessing/preprocessing_numerical_grouped_variables/concat.py

# Étape 2 : Preprocessing ordinales + catégorielles
python preprocessing/tests/run_preprocessing_on_data.py
```

---

## Vérification des résultats

Un notebook de test est disponible pour vérifier la qualité du preprocessing :

```
preprocessing/tests/analysis_X_train_preprocessed.ipynb
```

Ce notebook permet de vérifier :
- Absence de valeurs manquantes
- Distribution des variables
- Cohérence des transformations
- Statistiques descriptives

---

## Structure des fichiers

```
Code_Hickathon/
├── data/
│   ├── X_train.csv                                    # Données brutes (INPUT)
│   ├── X_numerical_grouped_cleaned_train.csv          # Sortie Étape 1
│   └── X_train_preprocessed_YYYYMMDD_HHMMSS.csv      # Sortie Étape 2 (FINALE)
│
├── preprocessing/
│   ├── preprocessing_numerical_grouped_variables/
│   │   ├── load.py                                    # Chargement des données
│   │   ├── numerical.py                               # Preprocessing numérique
│   │   ├── grouped.py                                 # Preprocessing groupement
│   │   └── concat.py                                  # SCRIPT ÉTAPE 1
│   │
│   ├── tests/
│   │   ├── run_preprocessing_on_data.py               # SCRIPT ÉTAPE 2
│   │   └── analysis_X_train_preprocessed.ipynb        # Notebook de vérification
│   │
│   ├── classes/
│   │   ├── ordinal_preprocessor.py                    # Classe preprocessing ordinal
│   │   └── categorical_preprocessor.py                # Classe preprocessing catégoriel
│   │
│   └── utils/
│       └── preprocessing_utils.py                     # Fonctions utilitaires
```

---

## Résumé

| Étape | Script | Input | Output | Temps estimé |
|-------|--------|-------|--------|--------------|
| 1 | `concat.py` | `X_train.csv` | `X_numerical_grouped_cleaned_train.csv` | Quelques minutes |
| 2 | `run_preprocessing_on_data.py` | `X_numerical_grouped_cleaned_train.csv` | `X_train_preprocessed_{timestamp}.csv` | Quelques minutes |

Le fichier final `X_train_preprocessed_{timestamp}.csv` est prêt pour l'entraînement des modèles.
