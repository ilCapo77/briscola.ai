# Piano operativo — Briscola AI

> Questo file è VOLUTAMENTE breve: fotografa lo stato reale e le prossime azioni.
> I dettagli storici vivono nei messaggi di commit, in `docs/plans/` (ipotesi, gate e
> risultati) e nel diario di bordo (<https://ai.briscola.dev/diario> + `docs/diario/`).

## Stato Corrente

- Versione: `0.23.1`. In preparazione `0.24.0` (in revisione maintainer): diario di bordo
  sul sito (`/diario`, fonte `static/diario.md` + approfondimenti `docs/diario/`), default
  UI = `bc_model_pimc_belief_64x10`, persistenza selezioni in localStorage.
- Produzione: <https://ai.briscola.dev> (FastAPI Cloud, stato su Redis, realtime pub/sub,
  event log Postgres in modalità `dataset` con eventi `ai_action` auditabili).
- Modello consigliato: `best_a2c_v8.npz` (encoder **v4** con memoria delle prese,
  `feature_dim=369`, hidden 256 via Net2Net, guard anti-overkill ON). Promosso v0.22.0:
  +0.89 su v7 (CI coppie +0.74..+1.05, big 100k).
- Avversari avanzati selezionabili: **`bc_model_pimc_belief_64x10`** (il più forte: PIMC 64
  determinizzazioni pesate dalla belief, finestra 10 — +3.66 su v8+solver, CI +3.32..+4.00,
  ~75 ms/mossa pensata; release v0.23.0), `bc_model_value_lookahead_8x8` (+2.12, più
  leggero), `bc_model_pimc_16x8`.
- Anti-cheat: agenti e modelli ricevono solo `PlayerObservation`, mai `GameState` completo.
  La vista full-state di debug è opt-in (`BRISCOLA_DEBUG_STATE_ENDPOINT=1`), 403 di default.
- CI GitHub Actions: ruff format/check, mypy, pytest+coverage su ogni push/PR. Lezione
  operativa: la cache Numba locale può mascherare firme rotte — la CI (compilazione fredda)
  è il gate di verità per i kernel.
- Statistica seat-fair: le CI si calcolano sull'unità COPPIA (stesso mazzo, seat scambiati)
  via `seat_fair_avg_point_diff_ci`/`seat_fair_score_rate_ci`. Le CI per-partita storiche
  erano anti-conservative.
- Test-àncora anti-divergenza: tabelle carte e `who_wins_trick` derivate dal dominio
  (`ai/card_tables.py`, `ai/trick_kernel.py`), parità dominio↔fast↔numba per encoder
  v1–v4 su partite specchiate, reward shaping JIT ↔ canonico.
- Diagnostica deploy: `/version` espone `recommended_model_present`,
  `value_lookahead_model_present`, `pimc_belief_model_present` e lo stato dell'event log.
- Artefatti locali (`data/`, `benchmarks/`) gitignored. Pulizia 2026-07-03: rimossi ~11.5 GB
  di dataset di rami chiusi (teacher v3, diagnostiche distillazione, dataset ExIt iter-0,
  value full-game) — tutti rigenerabili dagli script; ricette nei commit.
- Test rapidi: `pytest -m "not slow and not numba"` (~3s).

## Decisioni Chiuse

Ogni riga è una decisione con evidenza; numeri e diagnosi in
`docs/plans/belief-expert-iteration.md` (§ indicati), `docs/diario/*.md` e nei commit.

- **v8 è il default `.npz`** (v0.22.0): le feature v4 valgono +0.27 netto a parità di tutto
  (prima evidenza runtime positiva del programma encoder); la capacità (Net2Net 128→256) è
  marginale da sola. Catena completa nel piano §6 e in `docs/diario/05-catena-campioni.md`.
- **La search insegna facendo sparring, non facendosi copiare**: la distillazione
  PIMC→policy è stata chiusa NEGATIVA due volte (giugno su v6; luglio come ExIt iter-0) con
  diagnosi: la CE su mosse-argmax è lossy (hidden 512: ~100% train, 56–57% validation).
  v7 e v8 nascono dall'operatore che funziona: A2C con la search come avversario.
- **La belief network**: NO come input della policy (−0.56: ridondante, erode l'istinto);
  SÌ per pesare le determinizzazioni della search (+3.66 in produzione, v0.23.0). §5–6.
- **Niente expert full-game via value depth-1**: dose-risposta pulita (finestra 8: +1.80 →
  tutta partita: −5.16); a inizio partita il valore è dominato dal caso del mazzo (MAE ~14
  irriducibile) e l'argmax su V rumorosa perde dall'istinto della policy. §6.
- **Value decision-aligned**: chiuso negativo (il fit pairwise/leaf non migliora il ranking
  che conta per la lookahead).
- **Population league**: declassata (costo alto, beneficio non dimostrato vs league semplice).
- **Solver endgame esatto**: deployabile e in produzione in tutti gli agenti avanzati
  (+1.8/+1.9 sopra policy forti, anti-cheat via ricostruzione da `PlayerObservation`).

## Prossime Azioni

### 1. Release 0.24.0 (in revisione)

Diario di bordo pubblico, default PIMC-belief, persistenza selezioni UI. Post-deploy:
monitorare la CPU delle repliche nei primi giorni — il default costa ~50–100× per mossa
rispetto a v8 puro; se il traffico lo rende oneroso, tornare a default `bc_model` è un
one-liner nel frontend.

### 2. Monitoraggio Produzione E Audit

- Classificare le mosse sospette dal `decision_type` dell'event log: `fallback` → policy
  base; `solver` → ricostruzione/endgame; `lookahead`/`search` → value/PIMC/guard.
- Errori ricorrenti → nuova ipotesi misurabile, non fix estemporanei.
- Non usare dati umani per training finché volume, consenso, qualità e privacy non sono
  riverificati.

### 3. Hardening Continuo

Aggiungere test solo su casi reali sospetti o quando si toccano regole/observation/search.
Debito minore residuo (nessuno urgente; prenderlo quando si tocca l'area):

- DX script: `scripts/_common.py` condiviso e `logging` al posto di `print`;
- backend: test e2e WebSocket attraverso l'app; gestione riconnessione Redis nello store;
- AI: RNG serial vs parallel non riproducibile cross-`workers` in `decision_quality`;
- frontend: zero test JS; CSS monolitico (~900 righe).

### 4. Nuovo Ciclo Di Training Solo Con Nuova Ipotesi

Tutte le leve lato allievo e lato maestro-stimato sono state misurate e chiuse (v. Decisioni).

**Kernel PIMC JIT (2026-07-03, v0.25.0)**: la search PIMC belief è stata portata su Numba
(`ai/numba/pimc.py`): forza equivalente al python (+3.38 vs +3.83 contro lo stesso
controllo, CI sovrapposte), CPU **~2× più economica** per mossa (37ms vs 73ms) — non i
20-50× sperati, perché il costo python era già dominato dalle matmul BLAS di numpy.
La variante di produzione usa la search JIT. Conseguenza per PIMC-as-teacher: a ~37ms/mossa
resta impraticabile come opponent di training su milioni di partite a 64 det (config ridotte
tipo 16×8 sarebbero ~fattibili overnight, con edge maestro però più piccolo). Il ramo resta
in frigo con questa quantificazione. PPO/GAE: bassa priorità, nessuna evidenza che serva.

## Comandi Utili

Quality gate:

```bash
uv run ruff format src tests scripts
uv run ruff check --fix src tests scripts
uv run mypy src
uv run pytest
```

Report/event log:

```bash
uv run python scripts/report_event_log.py --db path/to/events.sqlite3
uv run python scripts/audit_event_log_games.py --db path/to/events.sqlite3 --json
uv run python scripts/export_ai_actions.py --db path/to/events.sqlite3 --out data/ai_actions.jsonl
```

Report modelli:

```bash
uv run python scripts/build_model_report.py
```

Avvio locale:

```bash
briscola-server --reload
```
