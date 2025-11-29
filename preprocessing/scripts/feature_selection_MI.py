"""
Script de feature selection par Mutual Information.

Utilise la Mutual Information pour capturer les relations non-linéaires entre features et target.

Output: Dataset avec ~30 features sélectionnées + rapport JSON
"""

import sys
import os
from datetime import datetime
from typing import Tuple, List
import json

# Ajouter le chemin parent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import polars as pl
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def load_preprocessed_data(data_path: str, target_path: str = None) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Charger les données préprocessées et la target.

    Args:
        data_path: Chemin vers le fichier CSV préprocessé (features)
        target_path: Chemin vers le fichier CSV contenant MathScore (y_train.csv)

    Returns:
        Tuple (X, y) où X est le DataFrame des features et y la target
    """
    print("📂 Chargement des données préprocessées...")
    X = pl.read_csv(data_path)

    # Supprimer colonne vide/sans nom si présente (bug de concat)
    if '' in X.columns:
        print("   ⚠️  Suppression de la colonne vide/sans nom")
        X = X.drop('')

    # Vérifier si MathScore est déjà présent dans X
    if 'MathScore' in X.columns:
        print("   ✅ MathScore trouvé dans le fichier préprocessé")
        y = X.select('MathScore').to_series()
        X = X.drop('MathScore')
    else:
        # Charger MathScore depuis y_train.csv
        if target_path is None:
            data_dir = os.path.dirname(data_path)
            target_path = os.path.join(data_dir, 'y_train.csv')

        print(f"   📂 Chargement de MathScore depuis: {os.path.basename(target_path)}")
        y_df = pl.read_csv(target_path)

        # Vérifier que les dimensions correspondent
        if len(y_df) != len(X):
            raise ValueError(f"Dimensions incompatibles: X={len(X)} lignes, y={len(y_df)} lignes")

        # Extraire la série MathScore
        if 'MathScore' in y_df.columns:
            y = y_df.select('MathScore').to_series()
        else:
            # Si le fichier n'a qu'une colonne sans nom ou nommée autrement
            y = y_df.to_series(0)

        print(f"   ✅ MathScore chargé avec succès")

    print(f"   Shape X: {X.shape}")
    print(f"   Shape y: {y.shape}")
    print(f"   Nombre de features: {len(X.columns)}")

    return X, y


def split_data(X: pl.DataFrame, y: pl.Series,
               test_size: float = 0.2,
               val_size: float = 0.2,
               random_state: int = 42) -> Tuple:
    """
    Split stratifié 60/20/20 sur bins de target.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion du test set
        val_size: Proportion du validation set
        random_state: Seed pour reproductibilité

    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n📊 Split des données (60/20/20)...")

    # Créer bins pour stratification
    n_bins = 10
    y_np = y.to_numpy()
    bins = np.percentile(y_np, np.linspace(0, 100, n_bins + 1))
    bins[-1] += 1  # Inclure la valeur max
    y_binned = np.digitize(y_np, bins[:-1])

    # Combiner X et y pour le shuffle
    df_combined = X.with_columns(pl.Series('MathScore', y_np))
    df_combined = df_combined.with_columns(pl.Series('_bin', y_binned))

    # Shuffle avec seed
    np.random.seed(random_state)
    df_shuffled = df_combined.sample(fraction=1.0, shuffle=True, seed=random_state)

    # Split stratifié manuel
    n_total = len(df_shuffled)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val

    # Sort by bin puis split
    df_sorted = df_shuffled.sort('_bin')

    train_indices = []
    val_indices = []
    test_indices = []

    for bin_val in range(1, n_bins + 1):
        bin_mask = df_sorted.select(pl.col('_bin') == bin_val).to_series()
        bin_indices = np.where(bin_mask.to_numpy())[0]

        n_bin = len(bin_indices)
        n_bin_test = int(n_bin * test_size)
        n_bin_val = int(n_bin * val_size)

        test_indices.extend(bin_indices[:n_bin_test])
        val_indices.extend(bin_indices[n_bin_test:n_bin_test + n_bin_val])
        train_indices.extend(bin_indices[n_bin_test + n_bin_val:])

    # Extraire les splits
    df_train = df_sorted[train_indices]
    df_val = df_sorted[val_indices]
    df_test = df_sorted[test_indices]

    # Séparer X et y
    X_train = df_train.drop(['MathScore', '_bin'])
    y_train = df_train.select('MathScore').to_series()

    X_val = df_val.drop(['MathScore', '_bin'])
    y_val = df_val.select('MathScore').to_series()

    X_test = df_test.drop(['MathScore', '_bin'])
    y_test = df_test.select('MathScore').to_series()

    print(f"   Train: {X_train.shape} ({len(X_train)/n_total*100:.1f}%)")
    print(f"   Val:   {X_val.shape} ({len(X_val)/n_total*100:.1f}%)")
    print(f"   Test:  {X_test.shape} ({len(X_test)/n_total*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def mutual_information_selection(X_train: pl.DataFrame,
                                  y_train: pl.Series,
                                  n_features: int = 30) -> Tuple[List[str], np.ndarray]:
    """
    Sélectionner features par Mutual Information.

    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        n_features: Nombre de features à sélectionner

    Returns:
        Tuple (liste des features sélectionnées, scores MI)
    """
    print("\n🔍 Feature Selection - Mutual Information...")

    # Convertir en numpy pour sklearn
    X_np = X_train.to_numpy()
    y_np = y_train.to_numpy()
    feature_names = X_train.columns

    # Calculer MI scores
    print("   Calcul des scores MI...")
    mi_scores = mutual_info_regression(X_np, y_np, random_state=42, n_neighbors=5)

    # Trier par importance
    mi_ranking = np.argsort(mi_scores)[::-1]
    selected_indices = mi_ranking[:n_features]
    selected_features_mi = [feature_names[i] for i in selected_indices]

    print(f"   Top 10 features MI:")
    for i in range(min(10, len(selected_features_mi))):
        idx = mi_ranking[i]
        print(f"      {i+1}. {feature_names[idx]}: {mi_scores[idx]:.4f}")

    return selected_features_mi, mi_scores


def save_results(selected_features: List[str],
                 mi_scores: np.ndarray,
                 feature_names: List[str],
                 output_dir: str):
    """
    Sauvegarder les résultats de la feature selection MI.

    Args:
        selected_features: Liste des features sélectionnées
        mi_scores: Scores MI pour toutes les features
        feature_names: Noms de toutes les features
        output_dir: Répertoire de sortie
    """
    print("\n💾 Sauvegarde des résultats...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Sauvegarder liste des features sélectionnées (JSON)
    features_file = os.path.join(output_dir, f'selected_features_MI_{timestamp}.json')
    with open(features_file, 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'method': 'mutual_information',
            'timestamp': timestamp
        }, f, indent=2)
    print(f"   Features sélectionnées: {features_file}")

    # 2. Sauvegarder rapport complet (JSON)
    report_file = os.path.join(output_dir, f'feature_selection_MI_report_{timestamp}.json')

    # Créer mapping feature -> MI score
    mi_scores_dict = {feature_names[i]: float(mi_scores[i]) for i in range(len(feature_names))}

    report = {
        'timestamp': timestamp,
        'method': 'mutual_information',
        'n_features_original': len(feature_names),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'mi_scores': mi_scores_dict,
        'top_10_mi': sorted(mi_scores_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   Rapport complet: {report_file}")


def save_selected_data(X_train: pl.DataFrame,
                       X_val: pl.DataFrame,
                       X_test: pl.DataFrame,
                       y_train: pl.Series,
                       y_val: pl.Series,
                       y_test: pl.Series,
                       selected_features: List[str],
                       output_dir: str):
    """
    Sauvegarder les datasets avec features sélectionnées.

    Args:
        X_train, X_val, X_test: Features des 3 splits
        y_train, y_val, y_test: Targets des 3 splits
        selected_features: Liste des features à garder
        output_dir: Répertoire de sortie
    """
    print("\n💾 Sauvegarde des datasets avec features sélectionnées...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Filtrer les features
    X_train_selected = X_train.select(selected_features)
    X_val_selected = X_val.select(selected_features)
    X_test_selected = X_test.select(selected_features)

    # Ajouter la target
    df_train = X_train_selected.with_columns(y_train.alias('MathScore'))
    df_val = X_val_selected.with_columns(y_val.alias('MathScore'))
    df_test = X_test_selected.with_columns(y_test.alias('MathScore'))

    # Sauvegarder
    train_file = os.path.join(output_dir, f'X_train_MI_{timestamp}.csv')
    val_file = os.path.join(output_dir, f'X_val_MI_{timestamp}.csv')
    test_file = os.path.join(output_dir, f'X_test_MI_{timestamp}.csv')

    df_train.write_csv(train_file)
    df_val.write_csv(val_file)
    df_test.write_csv(test_file)

    print(f"   Train: {train_file} ({df_train.shape})")
    print(f"   Val:   {val_file} ({df_val.shape})")
    print(f"   Test:  {test_file} ({df_test.shape})")


def main():
    """Pipeline de feature selection par Mutual Information."""

    print("\n" + "="*80)
    print("FEATURE SELECTION - MUTUAL INFORMATION")
    print("="*80 + "\n")

    # Paramètres
    N_FEATURES = 30

    # Chemins
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')

    # Lister les fichiers X_train_preprocessed_*.csv
    import glob
    preprocessed_files = glob.glob(os.path.join(data_dir, 'X_train_preprocessed_*.csv'))

    if not preprocessed_files:
        raise FileNotFoundError(f"Aucun fichier X_train_preprocessed_*.csv trouvé dans {data_dir}")

    # Prendre le plus récent
    data_path = max(preprocessed_files, key=os.path.getctime)
    print(f"📂 Fichier détecté: {os.path.basename(data_path)}\n")

    output_dir = data_dir

    # 1. Charger données
    X, y = load_preprocessed_data(data_path)

    # 2. Split 60/20/20
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Mutual Information
    selected_features, mi_scores = mutual_information_selection(X_train, y_train, n_features=N_FEATURES)

    print(f"\n✅ Features sélectionnées: {len(selected_features)}")

    # 4. Sauvegarder résultats
    save_results(
        selected_features=selected_features,
        mi_scores=mi_scores,
        feature_names=X_train.columns,
        output_dir=output_dir
    )

    # 5. Sauvegarder datasets avec features sélectionnées
    save_selected_data(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        selected_features,
        output_dir
    )

    print("\n" + "="*80)
    print("✅ FEATURE SELECTION MI TERMINÉE AVEC SUCCÈS!")
    print("="*80 + "\n")

    return selected_features


if __name__ == '__main__':
    selected_features = main()
