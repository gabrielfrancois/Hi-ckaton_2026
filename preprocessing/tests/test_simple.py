"""
Test simple sans dépendances pour valider la logique des classes
"""

import sys
import os

# Ajouter le chemin parent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    """Test que les classes peuvent être importées"""
    try:
        from classes.ordinal_preprocessor import OrdinalPreprocessor
        from classes.categorical_preprocessor import CategoricalPreprocessor
        print("✅ Imports réussis")
        return True
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False


def test_ordinal_initialization():
    """Test initialisation OrdinalPreprocessor"""
    try:
        from classes.ordinal_preprocessor import OrdinalPreprocessor

        ordinal_vars = ['ST006', 'ST008', 'ST254']
        preprocessor = OrdinalPreprocessor(ordinal_vars=ordinal_vars)

        assert preprocessor.ordinal_vars == ordinal_vars
        assert preprocessor.is_fitted == False
        assert len(preprocessor.variables_to_drop) == 0

        print("✅ Test initialisation OrdinalPreprocessor réussi")
        return True
    except Exception as e:
        print(f"❌ Test initialisation OrdinalPreprocessor échoué: {e}")
        return False


def test_categorical_initialization():
    """Test initialisation CategoricalPreprocessor"""
    try:
        from classes.categorical_preprocessor import CategoricalPreprocessor

        categorical_vars = ['Gender', 'Country']
        preprocessor = CategoricalPreprocessor(categorical_vars=categorical_vars)

        assert preprocessor.categorical_vars == categorical_vars
        assert preprocessor.is_fitted == False

        print("✅ Test initialisation CategoricalPreprocessor réussi")
        return True
    except Exception as e:
        print(f"❌ Test initialisation CategoricalPreprocessor échoué: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTS SIMPLES - VALIDATION LOGIQUE")
    print("="*60 + "\n")

    results = []

    results.append(test_imports())
    results.append(test_ordinal_initialization())
    results.append(test_categorical_initialization())

    print("\n" + "="*60)
    print(f"RÉSULTAT: {sum(results)}/{len(results)} tests réussis")
    print("="*60 + "\n")

    if all(results):
        print("🎉 Tous les tests de base sont passés!")
        sys.exit(0)
    else:
        print("⚠️  Certains tests ont échoué")
        sys.exit(1)
