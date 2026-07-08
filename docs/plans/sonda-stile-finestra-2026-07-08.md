# Nota tecnica — Finestra PIMC e sensibilità allo stile (2026-07-08)

> Nota **interna** (non capitolo di diario). Data, metodo, numeri, decisioni chiuse.
> Origine: aneddoto partita `be30cdcc` evento 17031 (presa 15, deck=6, IA di mano apre
> Re di bastoni, poi l'umano controlla il finale con l'asso di briscola). Da lì due domande:
> (1) la finestra PIMC 8→10 aiuta? (2) il modello sa adattarsi allo stile dell'avversario?

## Metodo

- **Audit finestra**: variante eval-only `bc_model_pimc_belief_16x10` (stessa dose della
  16x8, `max_unknown_cards=10`), seat-fair `--seed-suite medium --num-games 10000` su
  `best_a2c_v11.npz`, contro `16x8` e `heuristic_trump_saver`. Artefatti in
  `benchmarks/experiments/pimc_window_16x10/` (gitignored).
- **Sensibilità allo stile** (sonda controfattuale): stati REALI da self-play del dominio
  (v11 nel seggio IA vs `mirror`/`heuristic_trump_saver`/`heuristic_v1`), filtrati a
  carico-non-briscola-vs-liscia da leader (tavolo vuoto, in mano ≥1 carico non-briscola E
  ≥1 liscia non-briscola). Per ogni stato: congelo l'osservazione, sovrascrivo l'intero
  blocco encoder v4 (idx 310..368) con un **profilo empirico** (media/mediana del blocco
  osservato contro ciascun avversario — valori che co-occorrono davvero) e misuro lo
  spostamento della massa softmax per bucket (carico_nb / carico_br / liscio_nb /
  briscola_bassa). Bootstrap CI 95% sul mean `ΔP_carico_nb = P(saver) − P(mirror)`.
  Controllo positivo: swap del blocco fase (inizio↔finale) — deve muoversi.
- **Analisi temporale**: distribuzione di `trick_index`, `deck_size`, contatori di stile
  e categoria dell'argmax live sugli stessi stati.
- Sonde in scratchpad (non ancora nel repo): `style_sensitivity3.py` (controfattuale),
  `style_temporal.py` (temporale). n≈11.6k stati (1000 partite/avversario).

## Risultati

**R1 — Finestra 8→10: chiusa negativa.** Seat-fair 16x10 vs 16x8: **−0.25 pt**, CI accoppiata
**[−0.45, −0.05]** (esclude zero → piccolo svantaggio reale). Vs `heuristic_trump_saver`:
16x8 **+16.06** vs 16x10 **+15.50** (finestra larga ~0.56 peggio, proprio contro il punitore
dei carichi). Confermato dallo storico `pimc_pareto_2000/d16_u8_vs_d16_u10` (+0.248 pro-8).
→ Nessun cambio default, nessun trigger, niente PIMC 256x come giudice.

**R2 — "Segno invertito" ritirato: era artefatto out-of-distribution.** Le prime sonde con
profili **costruiti a mano** davano ΔP_carico_nb **positivo** (segno sbagliato). Causa:
avevo impostato `cuts≈0.7` mentre il valore empirico è **≈0.07** (10× fuori scala) → rete
fuori distribuzione. Con **profili empirici** il segno è **corretto** (meno carico contro il
conservatore):

| contrasto (saver − mirror) | ΔP_carico_nb | bootstrap CI95 | argmax-flip | massa → |
|---|---|---|---|---|
| profili empirici — media | **−0.0083** | [−0.0090, −0.0075] | 1.1% | liscio_nb +0.0065 |
| profili empirici — mediana | **−0.0244** | [−0.0266, −0.0225] | 3.5% | liscio_nb +0.0184 |
| controllo positivo (fase) | +0.0563 | — | 19.6% | (sonda discrimina ✓) |

→ La policy usa lo stile nel **verso giusto** (carico_nb → liscio_nb contro il tagliatore),
ma **debolissimamente**: ~1–2.4 pp di massa, 1–3.5% di flip, contro 5.6 pp / 19.6% del cambio
di fase.

**R3 — Perché è debole: segnale raro, non timing (ipotesi "stati precoci" falsificata).**

| | trick_index (med) | deck_size (med) | cuts /10 (med) | aperture_carico /10 (med) |
|---|---|---|---|---|
| pool completo (n=11607) | 8 | 18 | 0.1 | 0.0 |
| argmax live = carico_nb (17.7%) | **14** | **6** | 0.1 | 0.0 |
| argmax live = liscio_nb (77.9%) | 7 | 20 | 0.0 | 0.0 |

- Il modello, quando ha carico + liscia, **guida la liscia nel 77.9%** dei casi; guida il
  carico solo nel **17.7%**, e **quasi solo in endgame** (deck≤6), dove è spesso legittimo.
- I contatori di stile restano **≈0 in tutta la partita**: tagli e aperture-di-carico sono
  **eventi rari**, quindi normalizzati /10 restano vicini a zero anche a metà-fine partita.
  Il segnale discriminante disponibile è genuinamente magro — meccanismo = rarità
  dell'evento, **non** "si decide troppo presto" (i carichi si guidano tardi).

## Decisioni chiuse

- **Finestra 8→10 / trigger / PIMC 256x giudice**: chiusi (R1).
- **Encoder v5 "per feature di stile"**: chiuso. v11 è già encoder v4 (`feature_dim=369`):
  le feature di stile e pericolo ci sono e vengono usate nel verso giusto (R2).
- **"Segno sbagliato"**: ritirato — artefatto OOD dei profili manuali (R2).
- **v13 potential-shaping**: NON ora. Al massimo esercizio didattico per una prudenza
  generale, con aspettativa di ranking **esplicitamente bassa**: il bias residuo è piccolo
  (17.7%, per lo più endgame) e il segnale su cui condizionare è raro (R3).

## Caveat

- I numeri R2/R3 vengono da sonde in scratchpad, non da artefatti stabili del repo: il metodo
  qui è sufficiente a ricomputarli; se servono ricomputabili, promuovere le sonde a `scripts/`.
- La sonda controfattuale congela il resto dello stato e scambia un blocco che nella realtà
  co-varia con esso: anche il risultato R2 (segno corretto, CI sotto zero) resta un
  **controfattuale empirico**, non una prova causale pulita.
- Profili = medie/mediane su stati; il contrasto empirico mirror↔saver è di piccola magnitudine.
