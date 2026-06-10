# Modelo campeão — competição de previsão de gols (Copa 2026)

Comparação no **holdout temporal** (treino `< 2024-01-01`: 11,314 jogos | teste `>= 2024-01-01`: 1,868 jogos). Eleição por **log-loss** (menor é melhor); desempate por Brier.

| Modelo | log-loss | Brier | MAE casa | MAE visit. | Acurácia V/E/D | Campeão |
|---|---|---|---|---|---|---|
| Poisson + Dixon-Coles | 0.8694 | 0.5110 | 1.0457 | 0.8814 | 60.4% | ✅ |
| Poisson GLM | 0.8697 | 0.5111 | 1.0457 | 0.8814 | 60.4% |  |
| LightGBM (Poisson) | 0.9388 | 0.5517 | 1.1079 | 0.9476 | 57.0% |  |

## Por que este modelo venceu

O **Poisson + Dixon-Coles** obteve o menor log-loss (0.8694), métrica que mede a *qualidade das probabilidades* — exatamente o que o Monte Carlo consome ao simular o torneio. 
A correção de empates (ρ = -0.01707) melhorou a calibração dos placares baixos (0-0, 1-1), onde a Poisson independente subestima empates — reduzindo o log-loss em 0.0% sobre o GLM puro, sem o risco de overfitting do boosting.

### Métricas — o que cada uma diz
- **log-loss** (primária): penaliza confiança alta em previsões erradas; sensível à calibração das probabilidades V/E/D.
- **Brier**: erro quadrático das probabilidades; apoio à log-loss.
- **MAE de gols**: erro médio do λ previsto vs. gols reais (mandante/visitante).
- **Acurácia V/E/D**: fração de resultados acertados — intuitiva, mas ignora a qualidade da probabilidade, por isso não decide a eleição.

> Gerado automaticamente por `src/treino.py`. Fonte: tabela `comparacao_modelos`.
