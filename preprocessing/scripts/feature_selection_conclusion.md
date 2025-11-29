# Feature Selection - Mutual Information

## Résultats

**Dataset**: 1,172,086 échantillons × 110 features
**Split**: 60/20/20 (Train/Val/Test)

## Top 10 Features (MI Score)

1. `math_q2_average_score`: 0.897
2. `math_q3_average_score`: 0.888
3. `math_q1_average_score`: 0.835
4. `math_q4_average_score`: 0.798
5. `Year`: 0.744
6. `math_q5_average_score`: 0.726
7. ` `: 0.706 (à supprimer - créé lors de la concaténation)
8. `math_q6_average_score`: 0.672
9. `math_q9_average_score`: 0.661
10. `math_q8_average_score`: 0.654

## Conclusions

- **Prédicteurs dominants**: Les scores scolaires trimestriels en maths (MI > 0.8) sont les meilleurs prédicteurs du score PISA
- **Effet temporel**: L'année a un impact significatif (0.744) - probablement dû aux variations de curriculum/difficulté
- **Performance attendue**: Avec ces features, R² > 0.9 attendu
- **Validation**: Pas de leakage car les notes scolaires précèdent le test PISA standardisé