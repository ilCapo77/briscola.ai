# Approfondimento — Solver, PIMC e la lezione della distillazione

**Capitolo del diario:** [Capitolo 7](https://ai.briscola.dev/diario) · **Periodo:** 22–30 giugno 2026

## Il solver endgame (`edf20a2`)

A mazzo vuoto l'informazione è completa (le carte avversarie si deducono): minimax esatto
con memoizzazione su `GameState`, ricostruito dalla sola `PlayerObservation` (anti-cheat).
Valore: **+1.83** anche sopra una policy forte (`86ce8ed`), +1.9 confermato su 100k.

## PIMC: pensare simulando (`c18e90f`)

Perfect Information Monte Carlo: campiona N "mondi" compatibili con l'osservazione, gioca
ogni mossa candidata in ciascun mondo fino in fondo, media. La config Pareto per il runtime
era 16 determinizzazioni × finestra 8 (`b1e821b`).

## La distillazione che fallisce (e insegna)

Tentativo: travasare le mosse PIMC nella policy con supervised learning.

| Fatto | Numero |
|---|---|
| Mosse analizzate | 175.000 |
| Correzioni search "forti e affidabili" | ~15.000 (8.6%) |
| MLP hidden=512: accuracy in training | ~100% (memorizza) |
| La stessa, in validation | **56–57%** (non generalizza) |
| Soft label T=2/5/10 | Nessun miglioramento oltre il tetto 57.3% |

Verdetto (`75db285`): la cross-entropy su mosse-argmax è un operatore lossy; copiare mosse
pensate non trasferisce il pensiero. La strada giusta: usare la search come **avversario di
training** (sparring), non come libro di testo — è così che nasce v7.
