"""
CategoricalPreprocessor - Classe pour le preprocessing des variables catégorielles PISA (avec Polars)
"""

import polars as pl
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
from typing import Tuple, List, Dict


class CategoricalPreprocessor:
    """
    Preprocessor pour variables catégorielles du dataset PISA.

    Gain net: -17 variables catégorielles (-10 métadonnées + -7 redondances)
    Impact massif: Regroupement ISCO (620 → 10 catégories) = -97% features
    """

    def __init__(self, categorical_vars: List[str] = None):
        """
        Initialise le preprocessor.

        Args:
            categorical_vars: Liste des noms de variables catégorielles
        """
        self.categorical_vars = categorical_vars or []
        self.variables_to_drop = []
        self.isco_mapping = {}
        self.rare_categories_mapping = {}
        self.binary_encoders = {}
        self.onehot_encoder = None
        self.frequency_encoders = {}
        self.mode_values = {}
        self.is_fitted = False

    def drop_metadata_variables(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Supprimer 10 métadonnées catégorielles (risque ZÉRO).

        Variables supprimées:
        - Options questionnaires: Option_CT, Option_FL, Option_ICTQ, Option_PQ,
          Option_TQ, Option_UH, Option_WBQ
        - Identifiants administratifs: CYC, NatCen, SUBNATIO

        Args:
            df: DataFrame Polars à transformer

        Returns:
            DataFrame sans les métadonnées
        """
        metadata_vars = [
            'Option_CT', 'Option_FL', 'Option_ICTQ', 'Option_PQ',
            'Option_TQ', 'Option_UH', 'Option_WBQ',
            'CYC', 'NatCen', 'SUBNATIO'
        ]

        vars_to_drop = [v for v in metadata_vars if v in df.columns]

        if vars_to_drop:
            df = df.drop(vars_to_drop)
            if not self.is_fitted:
                self.variables_to_drop.extend(vars_to_drop)

        return df

    def drop_redundant_variables(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Supprimer 7 redondances catégorielles.

        Variables supprimées:
        - LANGTEST_PAQ, LANGTEST_QQQ: Redondant avec LANGTEST_COG
        - ST003D03T: Birth Year (redondant avec AGE)
        - ST001D01T: Grade (redondant avec GRADE)
        - PA008: Doublon
        - PA162: Lecture parent (garder ST168)
        - OCOD3: Profession aspirée (faible valeur + haute cardinalité)

        Args:
            df: DataFrame Polars à transformer

        Returns:
            DataFrame sans les redondances
        """
        redundant_vars = [
            'LANGTEST_PAQ', 'LANGTEST_QQQ', 'ST003D03T', 'ST001D01T',
            'PA008', 'PA162', 'OCOD3'
        ]

        vars_to_drop = [v for v in redundant_vars if v in df.columns]

        if vars_to_drop:
            df = df.drop(vars_to_drop)
            if not self.is_fitted:
                self.variables_to_drop.extend(vars_to_drop)

        return df

    def group_isco_codes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Regrouper codes ISCO (620 → 10 catégories).

        Extraction du 1er chiffre du code ISCO-08:
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

        Impact: OCOD1 (620) + OCOD2 (620) → OCOD1_grouped (10) + OCOD2_grouped (10)
        Réduction: -1240 features potentielles

        Args:
            df: DataFrame Polars à transformer

        Returns:
            DataFrame avec codes ISCO regroupés
        """
        isco_vars = ['OCOD1', 'OCOD2']

        for var in isco_vars:
            if var in df.columns:
                grouped_var = f"{var}_grouped"

                # Extraire le premier chiffre avec Polars
                df = df.with_columns(
                    pl.col(var).cast(pl.Utf8).str.slice(0, 1).cast(pl.Int64, strict=False).alias(grouped_var)
                )
                df = df.drop(var)

                if not self.is_fitted:
                    self.isco_mapping[var] = grouped_var

        return df

    def impute_mode(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                    df_test: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Imputer avec mode (calculé sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test

        Returns:
            Tuple (df_train_imputed, df_val_imputed, df_test_imputed)
        """
        categorical_cols = [col for col in self.categorical_vars if col in df_train.columns]

        if not categorical_cols:
            return df_train, df_val, df_test

        for col in categorical_cols:
            # Calculer le mode sur train
            mode_series = df_train.select(pl.col(col).mode()).to_series()

            if len(mode_series) > 0 and mode_series[0] is not None:
                mode_value = mode_series[0]
                self.mode_values[col] = mode_value

                # Imputer avec fill_null
                df_train = df_train.with_columns(pl.col(col).fill_null(mode_value))
                df_val = df_val.with_columns(pl.col(col).fill_null(mode_value))
                df_test = df_test.with_columns(pl.col(col).fill_null(mode_value))

        return df_train, df_val, df_test

    def group_rare_categories(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                              df_test: pl.DataFrame, threshold: float = 0.01) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Regrouper catégories < 1% en 'Other' (fit sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test
            threshold: Seuil de fréquence (défaut 1%)

        Returns:
            Tuple (df_train_grouped, df_val_grouped, df_test_grouped)
        """
        categorical_cols = [col for col in self.categorical_vars if col in df_train.columns]

        if not categorical_cols:
            return df_train, df_val, df_test

        n_train = len(df_train)

        for col in categorical_cols:
            # Calculer fréquences sur train
            value_counts = df_train.group_by(col).agg(pl.count()).with_columns(
                (pl.col('count') / n_train).alias('freq')
            )

            # Identifier catégories rares
            rare_categories = value_counts.filter(pl.col('freq') < threshold).select(col).to_series().to_list()

            if rare_categories:
                self.rare_categories_mapping[col] = rare_categories

                # Remplacer par 'Other' avec when/then
                df_train = df_train.with_columns(
                    pl.when(pl.col(col).is_in(rare_categories))
                    .then(pl.lit('Other'))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
                df_val = df_val.with_columns(
                    pl.when(pl.col(col).is_in(rare_categories))
                    .then(pl.lit('Other'))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
                df_test = df_test.with_columns(
                    pl.when(pl.col(col).is_in(rare_categories))
                    .then(pl.lit('Other'))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        return df_train, df_val, df_test

    def encode_binary_variables(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                                df_test: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Encoder variables binaires en 0/1 (fit sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test

        Returns:
            Tuple (df_train_encoded, df_val_encoded, df_test_encoded)
        """
        categorical_cols = [col for col in self.categorical_vars if col in df_train.columns]

        if not categorical_cols:
            return df_train, df_val, df_test

        for col in categorical_cols:
            n_unique = df_train.select(pl.col(col).n_unique()).item()

            if n_unique == 2:
                # Convertir en numpy pour sklearn
                encoder = LabelEncoder()

                train_vals = df_train.select(col).to_series().cast(pl.Utf8).to_numpy()
                val_vals = df_val.select(col).to_series().cast(pl.Utf8).to_numpy()
                test_vals = df_test.select(col).to_series().cast(pl.Utf8).to_numpy()

                train_encoded = encoder.fit_transform(train_vals)
                val_encoded = encoder.transform(val_vals)
                test_encoded = encoder.transform(test_vals)

                # Remplacer dans Polars
                df_train = df_train.with_columns(pl.Series(col, train_encoded))
                df_val = df_val.with_columns(pl.Series(col, val_encoded))
                df_test = df_test.with_columns(pl.Series(col, test_encoded))

                self.binary_encoders[col] = encoder

        return df_train, df_val, df_test

    def onehot_encode_low_cardinality(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                                      df_test: pl.DataFrame, max_categories: int = 10) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        One-Hot encoding pour cardinalité ≤10 (fit sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test
            max_categories: Seuil de cardinalité max pour one-hot

        Returns:
            Tuple (df_train_encoded, df_val_encoded, df_test_encoded)
        """
        categorical_cols = [col for col in self.categorical_vars if col in df_train.columns]

        # Filtrer les variables non-binaires avec cardinalité <= max_categories
        low_cardinality_cols = [
            col for col in categorical_cols
            if col not in self.binary_encoders and 3 <= df_train.select(pl.col(col).n_unique()).item() <= max_categories
        ]

        if not low_cardinality_cols:
            return df_train, df_val, df_test

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Convertir en numpy pour sklearn
        train_vals = df_train.select(low_cardinality_cols).to_numpy().astype(str)
        val_vals = df_val.select(low_cardinality_cols).to_numpy().astype(str)
        test_vals = df_test.select(low_cardinality_cols).to_numpy().astype(str)

        train_encoded = encoder.fit_transform(train_vals)
        val_encoded = encoder.transform(val_vals)
        test_encoded = encoder.transform(test_vals)

        # Créer noms de features
        feature_names = encoder.get_feature_names_out(low_cardinality_cols)

        # Créer DataFrames Polars
        train_encoded_df = pl.DataFrame(train_encoded, schema={name: pl.Float64 for name in feature_names})
        val_encoded_df = pl.DataFrame(val_encoded, schema={name: pl.Float64 for name in feature_names})
        test_encoded_df = pl.DataFrame(test_encoded, schema={name: pl.Float64 for name in feature_names})

        # Concatener
        df_train = pl.concat([df_train.drop(low_cardinality_cols), train_encoded_df], how='horizontal')
        df_val = pl.concat([df_val.drop(low_cardinality_cols), val_encoded_df], how='horizontal')
        df_test = pl.concat([df_test.drop(low_cardinality_cols), test_encoded_df], how='horizontal')

        self.onehot_encoder = encoder

        return df_train, df_val, df_test

    def frequency_encode_high_cardinality(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                                         df_test: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Frequency encoding pour cardinalité >10 (fit sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test

        Returns:
            Tuple (df_train_encoded, df_val_encoded, df_test_encoded)
        """
        categorical_cols = [col for col in self.categorical_vars if col in df_train.columns]

        # Filtrer les variables avec cardinalité > 10 (non binaires, non one-hot)
        high_cardinality_cols = [
            col for col in categorical_cols
            if col not in self.binary_encoders and df_train.select(pl.col(col).n_unique()).item() > 10
        ]

        if not high_cardinality_cols:
            return df_train, df_val, df_test

        for col in high_cardinality_cols:
            # Calculer fréquences sur train
            n_train = len(df_train)
            freq_map = df_train.group_by(col).agg(pl.count()).with_columns(
                (pl.col('count') / n_train).alias('freq')
            ).select([col, 'freq'])

            # Convertir en dict pour stockage
            freq_dict = dict(zip(freq_map[col].to_list(), freq_map['freq'].to_list()))
            self.frequency_encoders[col] = freq_dict

            # Appliquer mapping avec join
            df_train = df_train.join(freq_map, on=col, how='left').drop(col).rename({'freq': col})
            df_val = df_val.join(freq_map, on=col, how='left').drop(col).rename({'freq': col})
            df_test = df_test.join(freq_map, on=col, how='left').drop(col).rename({'freq': col})

            # Remplir NaN avec 0
            df_train = df_train.with_columns(pl.col(col).fill_null(0))
            df_val = df_val.with_columns(pl.col(col).fill_null(0))
            df_test = df_test.with_columns(pl.col(col).fill_null(0))

        return df_train, df_val, df_test

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Pipeline complet pour train (nettoyage avant split).

        Args:
            df: DataFrame d'entraînement

        Returns:
            DataFrame transformé
        """
        df = self.drop_metadata_variables(df)
        df = self.drop_redundant_variables(df)
        df = self.group_isco_codes(df)

        for var in self.variables_to_drop:
            if var in self.categorical_vars:
                self.categorical_vars.remove(var)

        for original_var, grouped_var in self.isco_mapping.items():
            if original_var in self.categorical_vars:
                self.categorical_vars.remove(original_var)
            self.categorical_vars.append(grouped_var)

        self.is_fitted = True

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Appliquer transformations sur val/test.

        Args:
            df: DataFrame de validation ou test

        Returns:
            DataFrame transformé
        """
        if not self.is_fitted:
            raise ValueError("Le preprocessor doit être fitté avant de transformer")

        df = self.drop_metadata_variables(df)
        df = self.drop_redundant_variables(df)
        df = self.group_isco_codes(df)

        return df

    def save(self, filepath: str):
        """Sauvegarder preprocessor."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'CategoricalPreprocessor':
        """Charger preprocessor sauvegardé."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
