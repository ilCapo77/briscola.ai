# Approfondimento — La serie dei campioni (v1→v10)

**Capitolo del diario:** [Capitolo 6](https://ai.briscola.dev/diario)

## La serie dei campioni

(Versione completa e sempre aggiornata: [report Excel scaricabile](https://github.com/ilCapo77/briscola.ai/raw/master/docs/reports/model_progress.xlsx) — dashboard con curva di progressione.)

| Modello | Data | Ricetta | vs predecessore | vs heuristic_v1 |
|---|---|---|---|---|
| best_a2c (proto) | feb 2026 | A2C encoder v1, mix baseline | — | +9.71 → ~+12.69 |
| "v1" 5M seed19 | 5 giu | league 5M su Numba | +1.10 | +13.91 |
| v2 (seed48+50) | 8 giu | teacher h2 → BC → A2C encoder v2 | **+3.76** (il salto più grande) | +16.77 |
| v3 | 23 giu | encoder v3 (310 feat) + BC-anchor | +0.63 | +17.23 |
| v4 | 28 giu | league 1M da v3 | +0.36 | +17.50 |
| v5 | 28 giu | league 1M da v4 | +0.34 | +17.83 |
| v6 | 28 giu | scaling 5M da v5 | +0.46 | +18.40 |
| v7 | 1 lug | 5M vs opponent value-lookahead(v6) | **+2.27** | +18.73 |
| v8 | 3 lug | encoder v4 + Net2Net 256, 2× 5M vs VL(v7) | +0.89 | +17.61 |
| **v9** | 4 lug | super training 20M vs mix (VL(v8) 65% / specchio 15% / bar 20%) | **+0.97** | **+18.78** (record) |
| **v10** | 5 lug | 'definitivo' 30M vs cartellone completo (PIMC belief 25% / VL 35% / specchio 15% / bar 25%) | +0.66 | **+20.52** (record) |

## La crisi di fine giugno (`5c81eb4`)

La curva di scaling v5→v6 (1M/3M/5M → +0.03/+0.22/+0.46) mostrò costi crescenti, e un
paradosso accese il dubbio di **non transitività**: v3 batteva v2 di +0.008 punti medi ma
perdeva nel conteggio vittorie (48.433 vs 48.683 su 100k). Ne nacquero il round-robin con
intervalli di confidenza e la regola: *non avviare la generazione successiva per inerzia*.

Nota storica: v8 faceva +17.61 su heuristic_v1, MENO del +18.73 di v7 — non transitività di
stile, non regressione (il gate di promozione è l'head-to-head appaiato con CI).

La non-transitività è stata SANATA da v9: il 20% di euristiche nel suo mix di training recupera
gli exploit di stile (18.78, record) senza costare forza al vertice (+0.97 su v8).
