"""
OrdinalPreprocessor - Classe pour le preprocessing des variables ordinales PISA (avec Polars)
"""

import polars as pl
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle
from typing import Tuple, List


class OrdinalPreprocessor:
    """
    Preprocessor pour variables ordinales du dataset PISA.

    Gain net: -7 variables ordinales (-5 suppressions + -2 par scores composites)
    """

    def __init__(self, ordinal_vars: List[str] = None):
        """
        Initialise le preprocessor.

        Args:
            ordinal_vars: Liste des noms de variables ordinales
        """
        self.ordinal_vars = ordinal_vars or []
        self.variables_to_drop = []
        self.composite_scores = {}
        self.encoders = {}
        self.scaler = None
        self.knn_imputer = None
        self.winsorize_limits = None
        self.is_fitted = False

    def drop_redundant_variables(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Supprimer 5 variables ordinales redondantes.

        Variables supprimées:
        - ST005: Education mère (redondant avec ST006)
        - ST007: Education père (redondant avec ST008)
        - ST253, ST255, ST097: Redondants

        Args:
            df: DataFrame Polars à transformer

        Returns:
            DataFrame sans les variables redondantes
        """
        redundant_vars = ['ST005', 'ST007', 'ST253', 'ST255', 'ST097']
        vars_to_drop = [v for v in redundant_vars if v in df.columns]

        if vars_to_drop:
            df = df.drop(vars_to_drop)
            if not self.is_fitted:
                self.variables_to_drop.extend(vars_to_drop)

        return df

    def create_composite_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Créer scores composites pour variables mesurant le même construit.

        Scores créés:
        - Score_Support_Parental: moyenne(PA003, ST300)
        - Score_Support_Enseignant: moyenne(ST100, ST270)

        Args:
            df: DataFrame Polars à transformer

        Returns:
            DataFrame avec scores composites
        """
        composite_definitions = {
            'Score_Support_Parental': ['PA003', 'ST300'],
            'Score_Support_Enseignant': ['ST100', 'ST270']
        }

        for score_name, variables in composite_definitions.items():
            existing_vars = [v for v in variables if v in df.columns]

            if len(existing_vars) >= 2:
                # Créer le score composite (moyenne avec gestion des NaN)
                df = df.with_columns(
                    pl.mean_horizontal(existing_vars).alias(score_name)
                )
                df = df.drop(existing_vars)

                if not self.is_fitted:
                    self.composite_scores[score_name] = existing_vars

        return df

    def impute_median_simple(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Imputer avec médiane (version simple pour avant split).

        Args:
            df: DataFrame à imputer

        Returns:
            DataFrame imputé
        """
        ordinal_cols = [col for col in self.ordinal_vars if col in df.columns]

        if not ordinal_cols:
            return df

        for col in ordinal_cols:
            # Calculer la médiane
            median_value = df.select(pl.col(col).median()).item()

            if median_value is not None:
                # Imputer avec fill_null
                df = df.with_columns(pl.col(col).fill_null(median_value))
            else:
                # Si pas de médiane (colonne entièrement nulle), remplir avec 0
                df = df.with_columns(pl.col(col).fill_null(0))

        return df

    def impute_knn(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                   df_test: pl.DataFrame, k: int = 5) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Imputer valeurs manquantes avec KNN (fit sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test
            k: Nombre de voisins pour KNN

        Returns:
            Tuple (df_train_imputed, df_val_imputed, df_test_imputed)
        """
        ordinal_cols = [col for col in self.ordinal_vars if col in df_train.columns]

        if not ordinal_cols:
            return df_train, df_val, df_test

        self.knn_imputer = KNNImputer(n_neighbors=k)

        # Convertir en numpy pour sklearn
        train_imputed = self.knn_imputer.fit_transform(df_train.select(ordinal_cols).to_numpy())
        val_imputed = self.knn_imputer.transform(df_val.select(ordinal_cols).to_numpy())
        test_imputed = self.knn_imputer.transform(df_test.select(ordinal_cols).to_numpy())

        # Reconvertir en Polars et remplacer
        for i, col in enumerate(ordinal_cols):
            df_train = df_train.with_columns(pl.Series(col, train_imputed[:, i]))
            df_val = df_val.with_columns(pl.Series(col, val_imputed[:, i]))
            df_test = df_test.with_columns(pl.Series(col, test_imputed[:, i]))

        return df_train, df_val, df_test

    def winsorize_outliers(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                          df_test: pl.DataFrame, limits: List[float] = [0.01, 0.01]) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Winsorization au 99ème percentile (fit sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test
            limits: Limites [lower, upper]

        Returns:
            Tuple (df_train_winsorized, df_val_winsorized, df_test_winsorized)
        """
        ordinal_cols = [col for col in self.ordinal_vars if col in df_train.columns]

        if not ordinal_cols:
            return df_train, df_val, df_test

        self.winsorize_limits = {}

        for col in ordinal_cols:
            # Calculer les percentiles sur train
            lower_bound = df_train.select(pl.col(col).quantile(limits[0])).item()
            upper_bound = df_train.select(pl.col(col).quantile(1 - limits[1])).item()

            self.winsorize_limits[col] = (lower_bound, upper_bound)

            # Appliquer winsorization
            df_train = df_train.with_columns(
                pl.col(col).clip(lower_bound, upper_bound)
            )
            df_val = df_val.with_columns(
                pl.col(col).clip(lower_bound, upper_bound)
            )
            df_test = df_test.with_columns(
                pl.col(col).clip(lower_bound, upper_bound)
            )

        return df_train, df_val, df_test

    def encode_ordinal_variables(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                                df_test: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Encoder variables ordinales en préservant l'ordre (fit sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test

        Returns:
            Tuple (df_train_encoded, df_val_encoded, df_test_encoded)
        """
        ordinal_cols = [col for col in self.ordinal_vars if col in df_train.columns]

        if not ordinal_cols:
            return df_train, df_val, df_test

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Convertir en numpy pour sklearn
        train_encoded = encoder.fit_transform(df_train.select(ordinal_cols).to_numpy())
        val_encoded = encoder.transform(df_val.select(ordinal_cols).to_numpy())
        test_encoded = encoder.transform(df_test.select(ordinal_cols).to_numpy())

        # Reconvertir en Polars
        for i, col in enumerate(ordinal_cols):
            df_train = df_train.with_columns(pl.Series(col, train_encoded[:, i]))
            df_val = df_val.with_columns(pl.Series(col, val_encoded[:, i]))
            df_test = df_test.with_columns(pl.Series(col, test_encoded[:, i]))

        self.encoders['ordinal'] = encoder

        return df_train, df_val, df_test

    def standardize_variables(self, df_train: pl.DataFrame, df_val: pl.DataFrame,
                             df_test: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Standardiser variables ordinales (fit sur train).

        Args:
            df_train: DataFrame d'entraînement
            df_val: DataFrame de validation
            df_test: DataFrame de test

        Returns:
            Tuple (df_train_scaled, df_val_scaled, df_test_scaled)
        """
        ordinal_cols = [col for col in self.ordinal_vars if col in df_train.columns]

        if not ordinal_cols:
            return df_train, df_val, df_test

        self.scaler = StandardScaler()

        # Convertir en numpy pour sklearn
        train_scaled = self.scaler.fit_transform(df_train.select(ordinal_cols).to_numpy())
        val_scaled = self.scaler.transform(df_val.select(ordinal_cols).to_numpy())
        test_scaled = self.scaler.transform(df_test.select(ordinal_cols).to_numpy())

        # Reconvertir en Polars
        for i, col in enumerate(ordinal_cols):
            df_train = df_train.with_columns(pl.Series(col, train_scaled[:, i]))
            df_val = df_val.with_columns(pl.Series(col, val_scaled[:, i]))
            df_test = df_test.with_columns(pl.Series(col, test_scaled[:, i]))

        return df_train, df_val, df_test

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Pipeline complet pour train (nettoyage avant split).

        Args:
            df: DataFrame d'entraînement

        Returns:
            DataFrame transformé
        """
        df = self.drop_redundant_variables(df)
        df = self.create_composite_scores(df)

        for var in self.variables_to_drop:
            if var in self.ordinal_vars:
                self.ordinal_vars.remove(var)

        for composite_name, original_vars in self.composite_scores.items():
            for var in original_vars:
                if var in self.ordinal_vars:
                    self.ordinal_vars.remove(var)
            self.ordinal_vars.append(composite_name)

        # Imputation simple avant split (médiane)
        df = self.impute_median_simple(df)

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

        df = self.drop_redundant_variables(df)
        df = self.create_composite_scores(df)

        return df

    def save(self, filepath: str):
        """Sauvegarder preprocessor."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'OrdinalPreprocessor':
        """Charger preprocessor sauvegardé."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
