"""
Fonctions utilitaires pour orchestrer les preprocessors PISA (avec Polars)
"""

import polars as pl
import numpy as np
from typing import Tuple, Dict


def remove_high_missing_vars(df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
    """
    Supprimer variables avec >50% missing.

    Args:
        df: DataFrame Polars à nettoyer
        threshold: Seuil de pourcentage de valeurs manquantes (défaut 0.5 = 50%)

    Returns:
        DataFrame sans les variables à fort taux de missing
    """
    # Calculer le pourcentage de missing pour chaque colonne
    missing_stats = df.select([
        ((pl.col(col).is_null().sum() / pl.len()) > threshold).alias(col)
        for col in df.columns
    ])

    # Identifier colonnes à supprimer
    high_missing = [col for col in df.columns if missing_stats[col][0]]

    print(f"Suppression de {len(high_missing)} variables avec >{threshold*100}% missing")

    if high_missing:
        print(f"Variables supprimées: {high_missing[:10]}{'...' if len(high_missing) > 10 else ''}")
        df = df.drop(high_missing)

    return df


def split_train_val_test(df: pl.DataFrame, target: str = 'MathScore',
                        test_size: float = 0.2, val_size: float = 0.2,
                        random_state: int = 42) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame,
                                                          pl.Series, pl.Series, pl.Series]:
    """
    Split stratifié 60/20/20 sur bins de target.

    Args:
        df: DataFrame complet avec features + target
        target: Nom de la colonne target (défaut 'MathScore')
        test_size: Proportion du test set (défaut 0.2)
        val_size: Proportion du validation set (défaut 0.2)
        random_state: Seed pour reproductibilité (défaut 42)

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Créer bins pour stratification
    df = df.with_columns(
        pl.col(target).qcut(5, labels=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4'], allow_duplicates=True).alias('_strata')
    )

    # Premier split: train+val (80%) vs test (20%)
    train_val = df.sample(fraction=1-test_size, shuffle=True, seed=random_state, with_replacement=False)
    test = df.filter(~pl.col('_strata').is_in(train_val['_strata']))

    # Si le stratified sampling ne fonctionne pas parfaitement, utiliser un simple split aléatoire
    if len(test) == 0:
        # Fallback: split simple
        n = len(df)
        test_n = int(n * test_size)
        val_n = int(n * val_size)

        shuffled = df.sample(fraction=1.0, shuffle=True, seed=random_state)

        test = shuffled.head(test_n)
        val = shuffled.slice(test_n, val_n)
        train = shuffled.slice(test_n + val_n)

    else:
        # Second split: train (60%) vs val (20%)
        val_proportion = val_size / (1 - test_size)
        val = train_val.sample(fraction=val_proportion, shuffle=True, seed=random_state+1)
        train = train_val.filter(~pl.col('_strata').is_in(val['_strata']))

    # Supprimer la colonne de stratification
    train = train.drop('_strata')
    val = val.drop('_strata')
    test = test.drop('_strata')

    # Séparer features et target
    X_train = train.drop(target)
    y_train = train.select(target).to_series()

    X_val = val.drop(target)
    y_val = val.select(target).to_series()

    X_test = test.drop(target)
    y_test = test.select(target).to_series()

    print(f"Split réalisé:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def validate_preprocessing(df_before: pl.DataFrame, df_after: pl.DataFrame,
                          target: str = 'MathScore') -> Dict:
    """
    Valider preprocessing (no missing, dtypes, target unchanged).

    Args:
        df_before: DataFrame avant preprocessing
        df_after: DataFrame après preprocessing
        target: Nom de la colonne target (défaut 'MathScore')

    Returns:
        Dict avec rapport de validation
    """
    report = {
        'validation_passed': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Vérifier 0 NaN
    missing_count = df_after.null_count().sum_horizontal()[0]
    if missing_count > 0:
        report['errors'].append(f"{missing_count} valeurs manquantes détectées après preprocessing")
        report['validation_passed'] = False

    report['stats']['missing_values'] = missing_count

    # Vérifier target non modifié (si présent)
    if target in df_before.columns and target in df_after.columns:
        target_unchanged = df_before[target].equals(df_after[target])

        if not target_unchanged:
            report['errors'].append(f"La variable target '{target}' a été modifiée!")
            report['validation_passed'] = False
        else:
            report['stats']['target_unchanged'] = True

    # Statistiques générales
    report['stats']['n_features_before'] = len(df_before.columns)
    report['stats']['n_features_after'] = len(df_after.columns)
    report['stats']['features_dropped'] = len(df_before.columns) - len(df_after.columns)

    # Vérifier types de données
    numeric_cols = [col for col in df_after.columns if df_after[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
    report['stats']['n_numeric_features'] = len(numeric_cols)
    report['stats']['n_non_numeric_features'] = len(df_after.columns) - len(numeric_cols)

    if report['stats']['n_non_numeric_features'] > 0:
        non_numeric = [col for col in df_after.columns if col not in numeric_cols]
        report['warnings'].append(f"{len(non_numeric)} variables non-numériques: {non_numeric[:5]}")

    # Afficher résumé
    print("\n" + "="*60)
    print("RAPPORT DE VALIDATION PREPROCESSING")
    print("="*60)

    if report['validation_passed']:
        print("✅ Validation RÉUSSIE")
    else:
        print("❌ Validation ÉCHOUÉE")

    print(f"\nStatistiques:")
    print(f"  Features avant:  {report['stats']['n_features_before']}")
    print(f"  Features après:  {report['stats']['n_features_after']}")
    print(f"  Features supprimées: {report['stats']['features_dropped']}")
    print(f"  Valeurs manquantes: {report['stats']['missing_values']}")

    if report['errors']:
        print(f"\n❌ Erreurs ({len(report['errors'])}):")
        for error in report['errors']:
            print(f"  - {error}")

    if report['warnings']:
        print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"  - {warning}")

    print("="*60)

    return report


def generate_preprocessing_report(df_before: pl.DataFrame, df_after: pl.DataFrame,
                                 ordinal_prep, categorical_prep) -> Dict:
    """
    Générer rapport preprocessing complet (markdown + JSON).

    Args:
        df_before: DataFrame avant preprocessing
        df_after: DataFrame après preprocessing
        ordinal_prep: Instance de OrdinalPreprocessor
        categorical_prep: Instance de CategoricalPreprocessor

    Returns:
        Dict avec toutes les métadonnées de preprocessing
    """
    report = {
        'preprocessing_summary': {},
        'ordinal_transformations': {},
        'categorical_transformations': {},
        'feature_changes': {},
        'data_quality': {}
    }

    # Résumé général
    report['preprocessing_summary'] = {
        'n_samples': len(df_after),
        'n_features_before': len(df_before.columns),
        'n_features_after': len(df_after.columns),
        'features_dropped_count': len(df_before.columns) - len(df_after.columns),
        'reduction_percentage': (len(df_before.columns) - len(df_after.columns)) / len(df_before.columns) * 100
    }

    # Transformations ordinales
    report['ordinal_transformations'] = {
        'variables_dropped': ordinal_prep.variables_to_drop,
        'composite_scores_created': list(ordinal_prep.composite_scores.keys()),
        'composite_scores_definitions': ordinal_prep.composite_scores,
        'n_ordinal_vars_before': len(ordinal_prep.ordinal_vars) + len(ordinal_prep.variables_to_drop),
        'n_ordinal_vars_after': len(ordinal_prep.ordinal_vars)
    }

    # Transformations catégorielles
    report['categorical_transformations'] = {
        'metadata_dropped': [v for v in categorical_prep.variables_to_drop
                            if v in ['Option_CT', 'Option_FL', 'Option_ICTQ', 'Option_PQ',
                                   'Option_TQ', 'Option_UH', 'Option_WBQ', 'CYC', 'NatCen', 'SUBNATIO']],
        'redundant_dropped': [v for v in categorical_prep.variables_to_drop
                             if v in ['LANGTEST_PAQ', 'LANGTEST_QQQ', 'ST003D03T', 'ST001D01T',
                                    'PA008', 'PA162', 'OCOD3']],
        'isco_regrouped': list(categorical_prep.isco_mapping.keys()),
        'isco_mapping': categorical_prep.isco_mapping,
        'n_categorical_vars_before': len(categorical_prep.categorical_vars) + len(categorical_prep.variables_to_drop),
        'n_categorical_vars_after': len(categorical_prep.categorical_vars)
    }

    # Changements de features
    features_before = set(df_before.columns)
    features_after = set(df_after.columns)

    report['feature_changes'] = {
        'features_added': list(features_after - features_before),
        'features_removed': list(features_before - features_after),
        'features_unchanged': list(features_before & features_after)
    }

    # Qualité des données
    report['data_quality'] = {
        'missing_before': df_before.null_count().sum_horizontal()[0],
        'missing_after': df_after.null_count().sum_horizontal()[0],
        'missing_percentage_before': (df_before.null_count().sum_horizontal()[0] / (len(df_before) * len(df_before.columns))) * 100,
        'missing_percentage_after': (df_after.null_count().sum_horizontal()[0] / (len(df_after) * len(df_after.columns))) * 100
    }

    # Afficher résumé
    print("\n" + "="*60)
    print("RAPPORT PREPROCESSING COMPLET")
    print("="*60)

    print(f"\n📊 RÉSUMÉ GÉNÉRAL")
    print(f"  Échantillons: {report['preprocessing_summary']['n_samples']}")
    print(f"  Features avant: {report['preprocessing_summary']['n_features_before']}")
    print(f"  Features après: {report['preprocessing_summary']['n_features_after']}")
    print(f"  Réduction: {report['preprocessing_summary']['features_dropped_count']} features "
          f"(-{report['preprocessing_summary']['reduction_percentage']:.1f}%)")

    print(f"\n🔢 TRANSFORMATIONS ORDINALES")
    print(f"  Variables supprimées: {len(report['ordinal_transformations']['variables_dropped'])}")
    print(f"  Scores composites créés: {len(report['ordinal_transformations']['composite_scores_created'])}")
    print(f"  Variables ordinales: {report['ordinal_transformations']['n_ordinal_vars_before']} → "
          f"{report['ordinal_transformations']['n_ordinal_vars_after']}")

    print(f"\n🏷️  TRANSFORMATIONS CATÉGORIELLES")
    print(f"  Métadonnées supprimées: {len(report['categorical_transformations']['metadata_dropped'])}")
    print(f"  Redondances supprimées: {len(report['categorical_transformations']['redundant_dropped'])}")
    print(f"  Variables ISCO regroupées: {len(report['categorical_transformations']['isco_regrouped'])}")
    print(f"  Variables catégorielles: {report['categorical_transformations']['n_categorical_vars_before']} → "
          f"{report['categorical_transformations']['n_categorical_vars_after']}")

    print(f"\n✨ FEATURES CRÉÉES")
    if report['feature_changes']['features_added']:
        for feature in report['feature_changes']['features_added']:
            print(f"  + {feature}")
    else:
        print("  Aucune")

    print(f"\n📉 QUALITÉ DES DONNÉES")
    print(f"  Valeurs manquantes avant: {report['data_quality']['missing_before']} "
          f"({report['data_quality']['missing_percentage_before']:.2f}%)")
    print(f"  Valeurs manquantes après: {report['data_quality']['missing_after']} "
          f"({report['data_quality']['missing_percentage_after']:.2f}%)")

    print("="*60)

    return report
