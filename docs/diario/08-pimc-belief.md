# Approfondimento — Il sussurro trova casa: PIMC belief in produzione

**Capitolo del diario:** [Capitolo 9](https://ai.briscola.dev/diario) · **Periodo:** 3 luglio 2026

## La belief network

MLP 369→128→40 con sigmoid: dalle feature v4 (incluso il comportamento avversario nelle
prese passate) stima P(carta in mano avversaria) per ognuna delle 40 carte. Offline:
top-k recall 0.593 contro 0.399 dell'uniforme. Anti-cheat invariato: è inferenza da
informazione pubblica, non sbirciatina.

## Dove NON funzionava e dove funziona

| Uso della belief | Esito |
|---|---|
| Pesare determinizzazioni PIMC, finestra deploy 16×8 | +0.15 n.s. (finestra troppo piccola) |
| Input aggiuntivo della policy (409 feature) | −0.56: dannosa (ridondante) |
| **Pesare 64 determinizzazioni, finestra 10, search a rollout** | **+3.83 → confermato +3.66** |

## I numeri di release (4.000 partite seat-fair, stessi semi, vs v8+solver)

| Agente | Punti/partita | CI95 | Costo |
|---|---|---|---|
| `bc_model_pimc_belief_64x10` | **+3.66** | +3.32..+4.00 | 0.267 s/partita (~75 ms/mossa pensata) |
| `bc_model_value_lookahead_8x8` (precedente) | +2.12 | +1.83..+2.41 | 0.021 s/partita |

Per ogni mossa nella finestra, l'agente simula 64 mani avversarie pesate sul comportamento
osservato e gioca ogni continuazione fino in fondo: ~670 rollout per partita reale.
Release: v0.23.0 (asset belief con provisioning `BRISCOLA_BELIEF_MODEL_URL/SHA256`).
