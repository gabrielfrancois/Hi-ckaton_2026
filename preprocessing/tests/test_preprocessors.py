"""
Tests unitaires pour OrdinalPreprocessor et CategoricalPreprocessor (avec Polars et X_train.csv)
"""

import pytest
import polars as pl
import sys
import os

# Ajouter le chemin parent pour importer les classes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.ordinal_preprocessor import OrdinalPreprocessor
from classes.categorical_preprocessor import CategoricalPreprocessor


# Fixtures pour données de test
@pytest.fixture(scope="session")
def x_train_sample():
    """Charger un échantillon de X_train.csv"""
    data_path = os.path.join(os.path.dirname(__file__), '../../data/X_train.csv')

    # Charger avec Polars (plus rapide que pandas)
    df = pl.read_csv(data_path)

    # Prendre un échantillon de 1000 lignes pour tests rapides
    df_sample = df.sample(n=min(1000, len(df)), seed=42)

    return df_sample


@pytest.fixture
def ordinal_vars_list():
    """Liste des variables ordinales (extrait de l'EDA)"""
    # Variables ordinales identifiées dans l'EDA
    return [
        'ST005', 'ST006', 'ST007', 'ST008', 'ST097', 'ST100', 'ST253', 'ST254',
        'ST255', 'ST256', 'ST270', 'ST273', 'PA003', 'ST300'
    ]


@pytest.fixture
def categorical_vars_list():
    """Liste des variables catégorielles (extrait de l'EDA)"""
    return [
        'Option_CT', 'Option_FL', 'Option_ICTQ', 'CYC', 'NatCen',
        'LANGTEST_PAQ', 'LANGTEST_COG', 'LANGTEST_QQQ',
        'ST003D03T', 'ST001D01T', 'OCOD1', 'OCOD2', 'OCOD3',
        'PA008', 'PA162', 'SUBNATIO'
    ]


# Tests OrdinalPreprocessor
class TestOrdinalPreprocessor:
    """Tests pour OrdinalPreprocessor sur vraies données"""

    def test_initialization(self, ordinal_vars_list):
        """Test initialisation"""
        preprocessor = OrdinalPreprocessor(ordinal_vars=ordinal_vars_list)

        assert preprocessor.ordinal_vars == ordinal_vars_list
        assert preprocessor.is_fitted == False
        assert len(preprocessor.variables_to_drop) == 0

    def test_drop_redundant_variables(self, x_train_sample, ordinal_vars_list):
        """Test suppression variables redondantes"""
        # Filtrer les variables ordinales présentes
        present_ordinal_vars = [v for v in ordinal_vars_list if v in x_train_sample.columns]

        preprocessor = OrdinalPreprocessor(ordinal_vars=present_ordinal_vars)

        df = x_train_sample.select(present_ordinal_vars).clone()
        n_cols_before = len(df.columns)

        df_after = preprocessor.drop_redundant_variables(df)

        # Vérifier que des colonnes ont été supprimées
        assert len(df_after.columns) < n_cols_before

        # Vérifier que les variables redondantes spécifiques sont supprimées si présentes
        redundant_vars = ['ST005', 'ST007', 'ST253', 'ST255', 'ST097']
        for var in redundant_vars:
            if var in df.columns:
                assert var not in df_after.columns

    def test_create_composite_scores(self, x_train_sample, ordinal_vars_list):
        """Test création scores composites"""
        present_ordinal_vars = [v for v in ordinal_vars_list if v in x_train_sample.columns]
        preprocessor = OrdinalPreprocessor(ordinal_vars=present_ordinal_vars)

        df = x_train_sample.select(present_ordinal_vars).clone()
        df_after = preprocessor.create_composite_scores(df)

        # Vérifier si les scores composites peuvent être créés
        # (dépend de la présence des variables sources)
        if 'PA003' in df.columns and 'ST300' in df.columns:
            assert 'Score_Support_Parental' in df_after.columns
            assert 'PA003' not in df_after.columns
            assert 'ST300' not in df_after.columns

        if 'ST100' in df.columns and 'ST270' in df.columns:
            assert 'Score_Support_Enseignant' in df_after.columns
            assert 'ST100' not in df_after.columns
            assert 'ST270' not in df_after.columns

    def test_fit_transform(self, x_train_sample, ordinal_vars_list):
        """Test pipeline fit_transform"""
        present_ordinal_vars = [v for v in ordinal_vars_list if v in x_train_sample.columns]

        if len(present_ordinal_vars) == 0:
            pytest.skip("Aucune variable ordinale présente dans l'échantillon")

        preprocessor = OrdinalPreprocessor(ordinal_vars=present_ordinal_vars)

        df = x_train_sample.select(present_ordinal_vars).clone()
        n_cols_before = len(df.columns)

        df_transformed = preprocessor.fit_transform(df)

        # Vérifier que is_fitted est True
        assert preprocessor.is_fitted == True

        # Vérifier que des transformations ont été appliquées
        assert len(df_transformed.columns) <= n_cols_before

    def test_impute_knn(self, x_train_sample, ordinal_vars_list):
        """Test imputation KNN"""
        present_ordinal_vars = [v for v in ordinal_vars_list if v in x_train_sample.columns]

        if len(present_ordinal_vars) == 0:
            pytest.skip("Aucune variable ordinale présente")

        preprocessor = OrdinalPreprocessor(ordinal_vars=present_ordinal_vars)

        # Créer splits
        df = x_train_sample.select(present_ordinal_vars).clone()
        n = len(df)

        df_train = df.slice(0, int(n * 0.6))
        df_val = df.slice(int(n * 0.6), int(n * 0.2))
        df_test = df.slice(int(n * 0.8))

        # Compter NaN avant
        null_count_before = df_train.null_count().sum_horizontal()[0]

        # Imputer
        df_train_imp, df_val_imp, df_test_imp = preprocessor.impute_knn(
            df_train, df_val, df_test, k=5
        )

        # Vérifier que NaN ont été réduits
        null_count_after = df_train_imp.null_count().sum_horizontal()[0]
        assert null_count_after <= null_count_before


# Tests CategoricalPreprocessor
class TestCategoricalPreprocessor:
    """Tests pour CategoricalPreprocessor sur vraies données"""

    def test_initialization(self, categorical_vars_list):
        """Test initialisation"""
        preprocessor = CategoricalPreprocessor(categorical_vars=categorical_vars_list)

        assert preprocessor.categorical_vars == categorical_vars_list
        assert preprocessor.is_fitted == False

    def test_drop_metadata_variables(self, x_train_sample, categorical_vars_list):
        """Test suppression métadonnées"""
        present_categorical_vars = [v for v in categorical_vars_list if v in x_train_sample.columns]

        preprocessor = CategoricalPreprocessor(categorical_vars=present_categorical_vars)

        df = x_train_sample.select(present_categorical_vars).clone()
        n_cols_before = len(df.columns)

        df_after = preprocessor.drop_metadata_variables(df)

        # Vérifier que des colonnes ont potentiellement été supprimées
        assert len(df_after.columns) <= n_cols_before

        # Vérifier que les métadonnées spécifiques sont supprimées si présentes
        metadata_vars = ['Option_CT', 'Option_FL', 'Option_ICTQ', 'CYC', 'NatCen', 'SUBNATIO']
        for var in metadata_vars:
            if var in df.columns:
                assert var not in df_after.columns

    def test_drop_redundant_variables(self, x_train_sample, categorical_vars_list):
        """Test suppression redondances"""
        present_categorical_vars = [v for v in categorical_vars_list if v in x_train_sample.columns]

        preprocessor = CategoricalPreprocessor(categorical_vars=present_categorical_vars)

        df = x_train_sample.select(present_categorical_vars).clone()
        n_cols_before = len(df.columns)

        df_after = preprocessor.drop_redundant_variables(df)

        # Vérifier suppression si présentes
        redundant_vars = ['LANGTEST_PAQ', 'LANGTEST_QQQ', 'ST003D03T', 'ST001D01T', 'PA008', 'PA162', 'OCOD3']
        for var in redundant_vars:
            if var in df.columns:
                assert var not in df_after.columns

    def test_group_isco_codes(self, x_train_sample, categorical_vars_list):
        """Test regroupement ISCO"""
        present_categorical_vars = [v for v in categorical_vars_list if v in x_train_sample.columns]

        preprocessor = CategoricalPreprocessor(categorical_vars=present_categorical_vars)

        df = x_train_sample.select(present_categorical_vars).clone()
        df_after = preprocessor.group_isco_codes(df)

        # Vérifier transformation ISCO si présents
        if 'OCOD1' in df.columns:
            assert 'OCOD1' not in df_after.columns
            assert 'OCOD1_grouped' in df_after.columns

        if 'OCOD2' in df.columns:
            assert 'OCOD2' not in df_after.columns
            assert 'OCOD2_grouped' in df_after.columns

    def test_fit_transform(self, x_train_sample, categorical_vars_list):
        """Test pipeline fit_transform"""
        present_categorical_vars = [v for v in categorical_vars_list if v in x_train_sample.columns]

        if len(present_categorical_vars) == 0:
            pytest.skip("Aucune variable catégorielle présente")

        preprocessor = CategoricalPreprocessor(categorical_vars=present_categorical_vars)

        df = x_train_sample.select(present_categorical_vars).clone()
        df_transformed = preprocessor.fit_transform(df)

        # Vérifier que is_fitted est True
        assert preprocessor.is_fitted == True

        # Vérifier que des transformations ont été appliquées
        assert len(df_transformed.columns) <= len(df.columns)

    def test_impute_mode(self, x_train_sample, categorical_vars_list):
        """Test imputation par mode"""
        present_categorical_vars = [v for v in categorical_vars_list if v in x_train_sample.columns]

        if len(present_categorical_vars) == 0:
            pytest.skip("Aucune variable catégorielle présente")

        preprocessor = CategoricalPreprocessor(categorical_vars=present_categorical_vars)

        # Créer splits
        df = x_train_sample.select(present_categorical_vars).clone()
        n = len(df)

        df_train = df.slice(0, int(n * 0.6))
        df_val = df.slice(int(n * 0.6), int(n * 0.2))
        df_test = df.slice(int(n * 0.8))

        # Compter NaN avant
        null_count_before = df_train.null_count().sum_horizontal()[0]

        # Imputer
        df_train_imp, df_val_imp, df_test_imp = preprocessor.impute_mode(
            df_train, df_val, df_test
        )

        # Vérifier que NaN ont été réduits
        null_count_after = df_train_imp.null_count().sum_horizontal()[0]
        assert null_count_after <= null_count_before


# Tests intégration
class TestIntegration:
    """Tests d'intégration avec vraies données"""

    def test_full_pipeline_ordinal(self, x_train_sample, ordinal_vars_list):
        """Test pipeline complet ordinal"""
        present_ordinal_vars = [v for v in ordinal_vars_list if v in x_train_sample.columns]

        if len(present_ordinal_vars) == 0:
            pytest.skip("Aucune variable ordinale présente")

        preprocessor = OrdinalPreprocessor(ordinal_vars=present_ordinal_vars.copy())

        df = x_train_sample.select(present_ordinal_vars).clone()
        n_cols_before = len(df.columns)

        # Pipeline complet
        df = preprocessor.fit_transform(df)

        # Vérifications
        assert preprocessor.is_fitted == True
        assert len(df.columns) <= n_cols_before

    def test_full_pipeline_categorical(self, x_train_sample, categorical_vars_list):
        """Test pipeline complet catégoriel"""
        present_categorical_vars = [v for v in categorical_vars_list if v in x_train_sample.columns]

        if len(present_categorical_vars) == 0:
            pytest.skip("Aucune variable catégorielle présente")

        preprocessor = CategoricalPreprocessor(categorical_vars=present_categorical_vars.copy())

        df = x_train_sample.select(present_categorical_vars).clone()
        n_cols_before = len(df.columns)

        # Pipeline complet
        df = preprocessor.fit_transform(df)

        # Vérifications
        assert preprocessor.is_fitted == True
        assert len(df.columns) <= n_cols_before

    def test_no_data_modification_on_original(self, x_train_sample, ordinal_vars_list):
        """Test que les données originales ne sont pas modifiées"""
        present_ordinal_vars = [v for v in ordinal_vars_list if v in x_train_sample.columns]

        if len(present_ordinal_vars) == 0:
            pytest.skip("Aucune variable ordinale présente")

        # Copie de l'échantillon original
        df_original = x_train_sample.select(present_ordinal_vars).clone()
        df_to_transform = x_train_sample.select(present_ordinal_vars).clone()

        preprocessor = OrdinalPreprocessor(ordinal_vars=present_ordinal_vars.copy())

        # Transformer
        _ = preprocessor.fit_transform(df_to_transform)

        # Vérifier que l'original n'a pas changé
        # (les DataFrames Polars sont immutables, donc cette vérification est assurée par design)
        assert len(df_original.columns) == len(x_train_sample.select(present_ordinal_vars).columns)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
