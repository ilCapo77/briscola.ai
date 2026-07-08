# Piano operativo — Briscola AI

> Questo file è volutamente breve: fotografa lo stato reale e le prossime azioni.
> La storia e i numeri completi vivono nei messaggi di commit, in `docs/plans/` e nel
> diario di bordo (<https://ai.briscola.dev/diario> + `docs/diario/`).

## Stato Corrente

- Versione live: `0.32.0` (2026-07-08), produzione su <https://ai.briscola.dev>.
- Runtime web: tutto Python, niente import Numba nel processo web. I kernel Numba restano
  per training, valutazioni e benchmark.
- Default UI: `bc_model_pimc_belief_16x8` su `best_a2c_v11.npz`, senza `overkill_guard`.
  La variante 64×10 resta selezionabile ma più costosa.
- Infrastruttura: FastAPI Cloud + Redis per stato/realtime + Postgres event log in modalità
  `dataset`. Scale-to-zero dopo ~90s di idle; avviso UI "server che si sveglia" live.
- Dati live: il log contiene `human_action`, `ai_action`, `game_finished`, metadati modello
  e consenso. `export_live_actions.py` produce un JSONL unico umano+IA; per i log vecchi
  inferisce la carta umana da `observation.my_hand[card_index]`, mentre le prossime partite
  avranno anche `result` esplicito su `human_action`.
- Diario: capitoli 13-16 pubblicati; approfondimenti tecnici 13-16 presenti in `docs/diario/`.
- Test rapidi: `pytest -m "not slow and not numba"` (~4s). Gate completo locale recente:
  ruff, mypy, pytest (`533 passed`).

## Prossima Decisione

Non lanciare altro training finché non c'è una nuova ipotesi misurata. v12 è chiusa:
forza quasi invariata su v11 e comportamento sui carichi praticamente identico.

Quando ci sono ~50-100 partite umane complete contro il default v11:

1. Esporta le azioni live filtrando bot/loadtest.
2. Conta carichi guidati dall'IA, carichi tagliati dagli umani, fase della partita e
   decision_type IA (`fallback`, `solver`, `search`, `lookahead`).
3. Misura anche cavata delle briscole con mano lunga, sprechi di briscola su piatti poveri
   e timing dell'asso di briscola.
4. Decidi il ramo:
   - **encoder v5 con feature di stile** se il problema è riconoscere il tipo di avversario;
   - **potential-shaping v13** se domina lo spreco di briscole;
   - **sonda PIMC mirata** se restano dubbi su cavata delle briscole o asso di briscola.

Gate minimo per encoder v5: meno carichi regalati contro il conservatore di briscole, ma
non prudenza generica contro lo specchio. Serve adattamento, non una regola rigida.

## Vincoli Operativi

- Anti-cheat: agenti e modelli ricevono solo `PlayerObservation`, mai `GameState` completo.
  La vista full-state è solo debug opt-in (`BRISCOLA_DEBUG_STATE_ENDPOINT=1`).
- Se cambia una regola o l'observation, mantenere allineati dominio canonico, fast path e
  Numba, con test di parità.
- Le valutazioni serie restano seat-fair a coppie: stesso mazzo, posti scambiati, CI sulle
  coppie (`seat_fair_avg_point_diff_ci`, `seat_fair_score_rate_ci`).
- La search insegna come avversario di sparring, non copiando le sue mosse. La distillazione
  PIMC→policy è già stata chiusa negativa.
- La belief network resta utile per pesare determinizzazioni PIMC, non come input diretto
  della policy.
- Non usare dati umani per training finché volume, consenso, qualità e privacy non sono
  rivalutati esplicitamente. Per ora i dati umani servono a orientare le ipotesi.
- Ogni release con bump versione: aggiornare `pyproject.toml`, rigenerare `uv.lock`, tag
  annotato, push tag, rigenerare `docs/reports/model_progress.xlsx` se la release cambia
  modelli o report ufficiali.

## Debito Aperto

- Audit campo v11: in attesa di abbastanza partite umane complete.
- Sonde da arbitrare con giudice PIMC: cavata delle briscole con mano lunga; asso di
  briscola giocato prima vs tenuto fino al finale.
- Infra: cold start residuo ~13.7s è piattaforma. Le uniche leve vere sono keep-alive <90s,
  piano a pagamento o ulteriore UX.
- Capacità default: free tier prudente ~5-8 giocatori simultanei. Se il traffico cresce,
  misurare una variante più economica prima di cambiare default.
- DX script: valutare `scripts/_common.py` e `logging` al posto di `print` quando si tocca
  ancora l'area export/audit.
- AI: RNG serial vs parallel non riproducibile cross-`workers` in `decision_quality`.
- Frontend: CSS monolitico (~900 righe), nessuna urgenza.

## Comandi Utili

Quality gate:

```bash
uv run ruff format src tests scripts
uv run ruff check --fix src tests scripts
uv run mypy src
uv run pytest
```

Audit/event log:

```bash
uv run python scripts/report_event_log.py --db path/to/events.sqlite3
uv run python scripts/audit_event_log_games.py --db path/to/events.sqlite3 --json
uv run python scripts/export_ai_actions.py --db path/to/events.sqlite3 --out data/ai_actions.jsonl
uv run python scripts/export_live_actions.py --db path/to/events.sqlite3 --out data/live_actions.jsonl
```

Export live consigliato per il prossimo audit:

```bash
DATABASE_URL=... uv run python scripts/export_live_actions.py \
  --ai-agent bc_model_pimc_belief_16x8 \
  --ai-model-id best_a2c_v11.npz \
  --exclude-client-id loadtest-bot \
  --out data/prod_live_actions_v11.jsonl
```

Profilo comportamentale locale:

```bash
uv run python scripts/behavior_profile.py \
  --model data/models/best_a2c_v11.npz \
  --opponents heuristic_trump_saver,mirror,heuristic_v1 \
  --num-games 2000
```

Report modelli:

```bash
uv run python scripts/build_model_report.py
```

Avvio locale:

```bash
briscola-server --reload
```
