"""
Script pour exécuter le preprocessing complet sur X_numerical_grouped_cleaned_train.csv et sauvegarder le résultat
"""

import sys
import os
from datetime import datetime

# Ajouter le chemin parent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import polars as pl
from classes.ordinal_preprocessor import OrdinalPreprocessor
from classes.categorical_preprocessor import CategoricalPreprocessor
from utils.preprocessing_utils import remove_high_missing_vars, generate_preprocessing_report


def load_variable_lists():
    """Définir les listes de variables ordinales et catégorielles"""

    # Variables ordinales (identifiées dans l'EDA)
    ordinal_vars = [
        'ST005', 'ST006', 'ST007', 'ST008', 'ST097', 'ST100', 'ST253', 'ST254',
        'ST255', 'ST256', 'ST270', 'ST273', 'PA003', 'ST300'
    ]

    # Variables catégorielles (identifiées dans l'EDA)
    categorical_vars = [
        'Option_CT', 'Option_FL', 'Option_ICTQ', 'Option_PQ', 'Option_TQ',
        'Option_UH', 'Option_WBQ', 'CYC', 'NatCen', 'SUBNATIO',
        'LANGTEST_PAQ', 'LANGTEST_COG', 'LANGTEST_QQQ',
        'ST003D03T', 'ST001D01T', 'OCOD1', 'OCOD2', 'OCOD3',
        'PA008', 'PA162'
    ]

    return ordinal_vars, categorical_vars


def impute_remaining_nulls(df: pl.DataFrame) -> pl.DataFrame:
    """
    Imputer toutes les colonnes restantes qui contiennent des valeurs manquantes.

    Stratégie:
    - Colonnes numériques (Float64, Int64): imputation par la médiane
    - Colonnes catégorielles (String, autres): imputation par le mode ou "Unknown"

    Args:
        df: DataFrame à imputer

    Returns:
        DataFrame sans valeurs manquantes
    """
    print("\n🔧 Imputation générique des valeurs manquantes restantes...")

    # Identifier les colonnes avec des nulls
    cols_with_nulls = [col for col in df.columns if df[col].null_count() > 0]

    if not cols_with_nulls:
        print("   ✅ Aucune valeur manquante détectée")
        return df

    print(f"   Colonnes avec NaNs: {len(cols_with_nulls)}")

    numeric_imputed = 0
    categorical_imputed = 0

    for col in cols_with_nulls:
        dtype = df[col].dtype
        null_count = df[col].null_count()

        # Colonnes numériques: imputation par médiane
        if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            median_value = df.select(pl.col(col).median()).item()

            if median_value is not None:
                df = df.with_columns(pl.col(col).fill_null(median_value))
                numeric_imputed += 1
            else:
                # Si pas de médiane (colonne entièrement nulle), remplir avec 0
                df = df.with_columns(pl.col(col).fill_null(0))
                numeric_imputed += 1

        # Colonnes catégorielles: imputation par mode
        else:
            mode_series = df.select(pl.col(col).mode()).to_series()

            if len(mode_series) > 0 and mode_series[0] is not None:
                mode_value = mode_series[0]
                df = df.with_columns(pl.col(col).fill_null(mode_value))
                categorical_imputed += 1
            else:
                # Si pas de mode, remplir avec "Unknown"
                df = df.with_columns(pl.col(col).cast(pl.Utf8).fill_null("Unknown"))
                categorical_imputed += 1

    print(f"   ✅ Imputation terminée:")
    print(f"      - {numeric_imputed} colonnes numériques (médiane)")
    print(f"      - {categorical_imputed} colonnes catégorielles (mode/Unknown)")

    # Vérification finale
    remaining_nulls = sum([df[col].null_count() for col in df.columns])
    if remaining_nulls == 0:
        print(f"   ✅ Plus aucune valeur manquante dans le dataset")
    else:
        print(f"   ⚠️  {remaining_nulls} valeurs manquantes restantes")

    return df


def preprocess_dataset(input_filename: str, output_prefix: str, ordinal_prep=None, categorical_prep=None, is_train=True):
    """
    Pipeline complet de preprocessing pour un dataset

    Args:
        input_filename: Nom du fichier d'entrée (ex: 'X_numerical_grouped_cleaned_train.csv')
        output_prefix: Préfixe pour le fichier de sortie (ex: 'X_train')
        ordinal_prep: Preprocessor ordinal pré-entraîné (None pour train, fitted pour test)
        categorical_prep: Preprocessor catégoriel pré-entraîné (None pour train, fitted pour test)
        is_train: True si c'est le dataset d'entraînement, False pour test

    Returns:
        tuple: (df_preprocessed, report, ordinal_prep, categorical_prep)
    """

    print("\n" + "="*80)
    print(f"PREPROCESSING COMPLET - {input_filename}")
    print("="*80 + "\n")

    # 1. Charger les données
    print(f"📂 Chargement de {input_filename}...")
    data_path = os.path.join(os.path.dirname(__file__), '../../data/', input_filename)
    df = pl.read_csv(data_path)
    df_original = df.clone()

    print(f"   Dimensions initiales: {df.shape}")
    print(f"   Colonnes: {len(df.columns)}")
    print(f"   Lignes: {len(df)}")

    # 2. Charger les listes de variables
    print("\n📋 Chargement des listes de variables...")
    ordinal_vars, categorical_vars = load_variable_lists()

    # Filtrer pour ne garder que les variables présentes
    ordinal_vars_present = [v for v in ordinal_vars if v in df.columns]
    categorical_vars_present = [v for v in categorical_vars if v in df.columns]

    print(f"   Variables ordinales présentes: {len(ordinal_vars_present)}/{len(ordinal_vars)}")
    print(f"   Variables catégorielles présentes: {len(categorical_vars_present)}/{len(categorical_vars)}")

    if is_train:
        # 3. Initialiser les preprocessors AVANT de supprimer les variables
        print("\n⚙️  Initialisation des preprocessors...")
        ordinal_prep = OrdinalPreprocessor(ordinal_vars=ordinal_vars_present.copy())
        categorical_prep = CategoricalPreprocessor(categorical_vars=categorical_vars_present.copy())

        # 4. Créer les scores composites AVANT de supprimer les variables à fort taux de missing
        print("\n🔢 Création des scores composites (avant suppression des variables)...")
        df = ordinal_prep.create_composite_scores(df)

        # Mettre à jour la liste des variables ordinales après création des composites
        for composite_name, original_vars in ordinal_prep.composite_scores.items():
            for var in original_vars:
                if var in ordinal_prep.ordinal_vars:
                    ordinal_prep.ordinal_vars.remove(var)
            ordinal_prep.ordinal_vars.append(composite_name)

        print(f"   Scores composites créés: {list(ordinal_prep.composite_scores.keys())}")

        # 5. Supprimer variables avec >50% missing (après création des composites)
        print("\n🧹 Suppression variables avec >50% missing...")
        df = remove_high_missing_vars(df, threshold=0.5)

        # Mettre à jour les listes
        ordinal_vars_present = [v for v in ordinal_prep.ordinal_vars if v in df.columns]
        categorical_vars_present = [v for v in categorical_vars_present if v in df.columns]

        # 6. Appliquer le reste du preprocessing ordinal
        print("\n🔢 Preprocessing des variables ordinales...")
        print("   - Suppression variables redondantes")
        df = ordinal_prep.drop_redundant_variables(df)

        # Mettre à jour la liste après suppression des redondantes
        for var in ordinal_prep.variables_to_drop:
            if var in ordinal_prep.ordinal_vars:
                ordinal_prep.ordinal_vars.remove(var)

        # Imputation médiane
        df = ordinal_prep.impute_median_simple(df)

        ordinal_prep.is_fitted = True

        print(f"   Variables ordinales après transformation: {len(ordinal_prep.ordinal_vars)}")
        print(f"   Variables supprimées: {ordinal_prep.variables_to_drop}")
        print(f"   Scores composites créés: {list(ordinal_prep.composite_scores.keys())}")

        # 7. Appliquer preprocessing catégoriel
        print("\n🏷️  Preprocessing des variables catégorielles...")
        print("   - Suppression métadonnées")
        print("   - Suppression redondances")
        print("   - Regroupement codes ISCO")
        df = categorical_prep.fit_transform(df)

        print(f"   Variables catégorielles après transformation: {len(categorical_prep.categorical_vars)}")
        print(f"   Variables supprimées: {categorical_prep.variables_to_drop}")
        print(f"   Regroupements ISCO: {categorical_prep.isco_mapping}")
    else:
        # Pour le test set, appliquer les transformations déjà apprises
        print("\n🔢 Application des transformations ordinales (depuis train)...")
        df = ordinal_prep.transform(df)

        print("\n🏷️  Application des transformations catégorielles (depuis train)...")
        df = categorical_prep.transform(df)

    # 8. Imputation générique des colonnes restantes
    df = impute_remaining_nulls(df)

    # 9. Générer rapport
    print("\n📊 Génération du rapport de preprocessing...")
    report = generate_preprocessing_report(df_original, df, ordinal_prep, categorical_prep)

    # 10. Sauvegarder le résultat
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_prefix}_preprocessed_{timestamp}.csv"
    output_path = os.path.join(os.path.dirname(__file__), '../../data/', output_filename)

    print(f"\n💾 Sauvegarde du résultat...")
    print(f"   Fichier: {output_filename}")
    df.write_csv(output_path)

    print(f"\n✅ Preprocessing terminé avec succès!")
    print(f"   Fichier sauvegardé: {output_path}")
    print(f"   Dimensions finales: {df.shape}")
    print(f"   Réduction: {df_original.shape[1] - df.shape[1]} colonnes supprimées")

    print("\n" + "="*80)

    return df, report, ordinal_prep, categorical_prep


def main():
    """Pipeline complet de preprocessing pour train et test"""

    # 1. Preprocessing du train set
    print("\n" + "🚂 " * 20)
    print("TRAIN SET")
    print("🚂 " * 20 + "\n")

    df_train, report_train, ordinal_prep, categorical_prep = preprocess_dataset(
        input_filename='X_numerical_grouped_cleaned_train.csv',
        output_prefix='X_train',
        is_train=True
    )

    # 2. Preprocessing du test set avec les mêmes transformations
    print("\n" + "🧪 " * 20)
    print("TEST SET")
    print("🧪 " * 20 + "\n")

    df_test, report_test, _, _ = preprocess_dataset(
        input_filename='X_numerical_grouped_cleaned_test.csv',
        output_prefix='X_test',
        ordinal_prep=ordinal_prep,
        categorical_prep=categorical_prep,
        is_train=False
    )

    return df_train, df_test, report_train, report_test


if __name__ == '__main__':
    df_train, df_test, report_train, report_test = main()
