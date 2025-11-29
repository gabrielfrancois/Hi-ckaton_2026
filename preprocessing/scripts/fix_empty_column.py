"""
Script pour supprimer la colonne vide/sans nom des fichiers preprocessed.

Corrige le bug de concat.py qui a créé une colonne vide en première position.
Utilise Polars pour un traitement rapide.
"""

import os
import sys
import glob

# Ajouter le chemin parent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import polars as pl


def fix_empty_column(file_path: str) -> bool:
    """
    Supprimer la colonne vide d'un fichier CSV et le réécrire en place.

    Args:
        file_path: Chemin vers le fichier CSV

    Returns:
        True si une colonne vide a été supprimée, False sinon
    """
    print(f"\n📂 Traitement: {os.path.basename(file_path)}")

    # Charger le fichier
    df = pl.read_csv(file_path)
    print(f"   Shape avant: {df.shape}")

    # Vérifier si colonne vide existe
    if '' in df.columns:
        print(f"   ⚠️  Colonne vide détectée - suppression...")
        df = df.drop('')

        # Réécrire le fichier (écrase l'original)
        df.write_csv(file_path)
        print(f"   ✅ Shape après: {df.shape}")
        print(f"   ✅ Fichier mis à jour")
        return True
    else:
        print(f"   ℹ️  Pas de colonne vide")
        return False


def main():
    """Parcourir tous les fichiers preprocessed et supprimer les colonnes vides."""

    print("\n" + "="*80)
    print("SUPPRESSION COLONNE VIDE - FICHIERS PREPROCESSED")
    print("="*80 + "\n")

    # Répertoire data
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')

    # Patterns de fichiers à traiter
    patterns = [
        'X_train_preprocessed_*.csv',
        'X_test_preprocessed_*.csv',
        'X_numerical_grouped_cleaned_train.csv',
        'X_numerical_grouped_cleaned_test.csv'
    ]

    files_processed = 0
    files_fixed = 0

    for pattern in patterns:
        matching_files = glob.glob(os.path.join(data_dir, pattern))

        for file_path in matching_files:
            files_processed += 1
            if fix_empty_column(file_path):
                files_fixed += 1

    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ")
    print("="*80)
    print(f"   Fichiers traités: {files_processed}")
    print(f"   Fichiers corrigés: {files_fixed}")
    print(f"   Fichiers OK: {files_processed - files_fixed}")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
