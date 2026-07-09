# Piano operativo — Briscola AI

> Questo file è volutamente breve: fotografa lo stato reale e le prossime azioni.
> La storia e i numeri completi vivono nei messaggi di commit, in `docs/plans/` e nel
> diario di bordo (<https://ai.briscola.dev/diario> + `docs/diario/`).

## Stato Corrente

- Versione live: `0.34.0` (2026-07-09), produzione su <https://ai.briscola.dev>. Include
  la promozione di `best_a2c_v13.npz` come default e l'audit di campo introdotto in 0.33.0:
  `human_action` salva il `result` + `export_live_actions.py` esporta un JSONL unico
  mosse umane+IA.
- Runtime web: tutto Python, niente import Numba nel processo web. I kernel Numba restano
  per training, valutazioni e benchmark.
- Default UI in preparazione: `bc_model_pimc_belief_16x8` su `best_a2c_v13.npz`, senza
  `overkill_guard`. La variante 64×10 resta selezionabile ma più costosa.
- Infrastruttura: FastAPI Cloud + Redis per stato/realtime + Postgres event log in modalità
  `dataset`. Scale-to-zero dopo ~90s di idle; avviso UI "server che si sveglia" live.
- Dati live: il log contiene `human_action`, `ai_action`, `game_finished`, metadati modello
  e consenso. `export_live_actions.py` produce un JSONL unico umano+IA; per i log vecchi
  inferisce la carta umana da `observation.my_hand[card_index]`, mentre le prossime partite
  avranno anche `result` esplicito su `human_action`.
- Audit finestra PIMC + sensibilità allo stile (2026-07-08, nota
  `docs/plans/sonda-stile-finestra-2026-07-08.md`): 8→10 secco chiuso negativo
  (`-0.25`, CI `-0.45..-0.05` vs 16x8; peggio anche vs trump_saver). v11 è già encoder v4
  (`feature_dim=369`) e usa le feature di stile nel **verso giusto ma debolmente** (il
  "segno sbagliato" iniziale era artefatto OOD dei profili manuali); il segnale di stile è
  raro (tagli/aperture-carico ≈0 per tutta la partita). Encoder v5 chiuso; v13 solo come
  esercizio didattico a payoff basso.
- v13 overkill shaping: `beta=0.3` a 5M è la prima promozione comportamentale riuscita.
  Non dimostra un salto di forza, ma mantiene v11 entro rumore: policy-only `-0.03`
  (CI `-0.38..+0.32`) e default PIMC 16x8 `+0.14` (CI `-0.20..+0.47`) contro v11.
  Riduce però nettamente l'overkill di briscola su piatti poveri: circa `28-31%` su v11
  → `6-8%` su v13 nei gate di qualità. Decisione: promuovere `best_a2c_v13.npz` come
  default `0.34.0` con wording onesto, "stessa forza, comportamento migliore".
- Diario: capitoli 13-16 pubblicati; approfondimenti tecnici 13-16 presenti in `docs/diario/`.
- Test rapidi: `pytest -m "not slow and not numba"` (~4s). Gate completo locale recente:
  ruff, mypy, pytest (`544 passed`).

## Prossima Decisione

Non lanciare altro training finché non c'è una nuova ipotesi misurata. v12 è chiusa:
forza quasi invariata su v11 e comportamento sui carichi praticamente identico.

**Le tre piste "comportamento sospetto vs umani" sono chiuse con evidenza**: carichi
guidati (sonda `lead_load_guard_probe.py`), timing dell'asso di briscola e cavata con mano
lunga (sonda `trump_play_probe.py`). Su questi assi v11 è già forte: nessun guard/shaping
migliora seat-fair. **Prossimo passo azionabile: verificare il deploy `0.34.0` (`/version`,
catalogo modelli, default UI) e poi monitorare il campo su v13**; l'audit resta gated dal
volume di partite umane.

Quando ci sono ~50-100 partite umane complete contro il default v13:

1. Esporta le azioni live filtrando bot/loadtest.
2. Conta carichi guidati dall'IA, carichi tagliati dagli umani, fase della partita e
   decision_type IA (`fallback`, `solver`, `search`, `lookahead`).
3. Misura anche cavata delle briscole con mano lunga, sprechi di briscola su piatti poveri
   e timing dell'asso di briscola.
4. Decidi il ramo:
   - **adattamento allo stile**: chiuso come pista forte (nota
     `docs/plans/sonda-stile-finestra-2026-07-08.md`). v11 usa già le feature v4 nel verso
     giusto ma debolmente, e il segnale di stile è raro. Non encoder v5, non opponent-modeling
     via feature;
   - **potential-shaping v13** se domina lo spreco di briscole; sul solo bias dei carichi
     guidati ha payoff basso (comportamento piccolo e per lo più endgame);
   - **lead-load guard / v13-bis** solo dopo una Fase 0 strumentata sui lead di carico v11:
     misurare volume, carichi persi/tagliati, fase, point_diff, presenza di liscia alternativa
     e rischio pubblico (`not_master`: esiste una carta ignota che potrebbe battere il carico).
     Nota: in Briscola non c'è obbligo di rispondere al seme, quindi "vuoto mostrato" non è
     una prova causale forte. Non usare trigger quasi vacui come `unknown_trumps_count >= 1`;
     se si testa un guard, deve essere eval-only, diagnostico, con sostituzione base via
     `card_conservation_cost`. Prime ablation 1000x/opponente: `not_master`,
     `thin_and_not_master` e variante `deck<=6 + avanti` riducono un po' i tagli ma perdono
     punti contro mirror/trump_saver; il guard runtime è improbabile, resta utile come sonda.
     Evidenza `by_deck_size` (v11 vs trump_saver, 1000 partite) che chiude anche l'ipotesi
     "intervenire a inizio partita": il modello **quasi non guida carichi presto**
     (lead_load_pct 4.35% con deck>16 vs 15.6% mid e 19.3% con deck<=6), quindi a inizio
     partita non c'è nulla da guardare. Il `cut_pct` è ~costante (63-66%) in tutte le fasi:
     è una scelta sistematica (aprire un carico per stanare la briscola), non un blunder
     localizzato — e il modello resta +14 pt vs il punitore *nonostante* i tagli. L'ablation
     `not_master` fa 201 interventi (scarta 2128 pt di carichi per lisce da 0) ma `cut_pct`
     scende solo -1.9 e il punteggio -1.08: il guard **rimanda** il taglio, non lo evita, e
     rimuove i sacrifici buoni (late il 28% dei carichi è già master e viene giustamente
     risparmiato). Conclusione: non è questione di *quando* far scattare il guard; nessun
     trigger lecito isola una perdita reale.
     **Cross-check su dati di campo** (`scripts/field_load_cut_base_rate.py`, default
     `bc_model_pimc_belief_16x8`): il "carico guidato tardi tagliato dall'umano" è un evento di
     *base rate*, non un marcatore di sconfitta. Attenzione al filtro data: l'export raw da 66
     partite mischiava 2026-07-08 (47) e 2026-07-09 (19) — separandole (deck<=8, load>=10):
     *07-09* (giorno degli aneddoti) perse 50% vs non-perse 46% → lift 1.08, **Fisher p=1.0**
     (nessun segnale); *07-08* perse 69% vs 41% → lift 1.68, p=0.11 (il debole segnale veniva
     quasi solo da qui); *mischiato* lift 1.48, p=0.18. In tutti i tagli, nelle partite dove
     l'evento capita **l'IA NON perde ~60-67% delle volte**. Gli aneddoti sulle singole partite
     perse sono reali ma è bias di selezione + probabile causazione inversa (l'IA in ritardo
     dumpa più carichi tardi): sintomo, non causa. Coerente con l'ablation controfattuale che
     *toglie* punti.
     **Barra per riaprire la pista** (decisione 2026-07-09): NON basta che con più partite il
     lift diventi statisticamente significativo — "evento più frequente nelle sconfitte" non
     implica "rimuoverlo migliora" (resta plausibile sintomo di partita già difficile). Serve
     un risultato più forte: una variante controfattuale che riduce l'evento **senza perdere
     punti** seat-fair. Quel test è già stato fatto e dice no. Rilanciare il base-rate solo
     con campione molto più grande (≥ qualche centinaio di partite live) o casi umani
     qualitativamente nuovi.
   - **cavata briscole / timing asso di briscola**: chiuso con `scripts/trump_play_probe.py`
     (Tier A descrittivo + ablation controfattuale seat-fair + cross-check regret endgame).
     v11, 1000 partite/avversario. *Asso*: il modello **non guida l'asso di briscola presto**
     (ace_led_early ≈4-9 su 1000; l'asso guidato è quasi solo in endgame `deck<=6`) e lo
     valorizza (avg_capture ~16-18, 43-63% delle prese ≥17 pt); il trattamento `ace_hold`
     fa 4-9 interventi/1000 → delta nullo, il fenomeno non esiste. *Cavata*: con ≥2 briscole
     il modello apre in briscola ~81% (≥3: 100%) e vince ~85%; `pull_more` è neutro
     (delta ±0.2), `pull_less` **crolla** (-6.7 vs trump_saver, -3.0 mirror, -5.3 v1). La
     cavata aggressiva con mano lunga è quindi una scelta buona *appresa*, non un bias.
     Tier C: esecuzione endgame quasi ottima (suboptimal 1.6-6.9%, regret media <0.6 pt).
     Conclusione: niente guard e niente shaping su asso/cavata (criterio Tier B fallito da
     tutti e tre i trattamenti).

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

- Audit campo v11: primo campione reale (66 partite live 2026-07-08+09; su 07-09: 19 partite,
  6 sconfitte IA). Base-rate confermato: il "carico guidato tardi tagliato" non è causa di
  sconfitta (vedi nota lead-load in Prossima Decisione, con numeri per-data). Metodo pronto in
  `scripts/field_load_cut_base_rate.py` (`--date`/`--timezone`). Raccogliere di più per assi
  non ancora coperti.
- Sonde cavata-briscole / asso di briscola: **arbitrate e chiuse** (2026-07-09,
  `scripts/trump_play_probe.py`) — il modello non guida l'asso presto e la cavata con mano
  lunga è corretta; niente guard/shaping. Dettaglio nella nota "Prossima Decisione".
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
  --ai-model-id best_a2c_v13.npz \
  --exclude-client-id loadtest-bot \
  --out data/prod_live_actions_v11.jsonl
```

Profilo comportamentale locale:

```bash
uv run python scripts/behavior_profile.py \
  --model data/models/best_a2c_v13.npz \
  --opponents heuristic_trump_saver,mirror,heuristic_v1 \
  --num-games 2000
```

Sonde diagnostiche comportamentali (chiuse, riproducibili):

```bash
# Carichi guidati: Fase 0 + ablation guard eval-only
uv run python scripts/lead_load_guard_probe.py --mode guard --num-games 1000
# Asso di briscola + cavata con mano lunga: Fase 0 + ablation controfattuale + regret endgame
uv run python scripts/trump_play_probe.py --mode both --num-games 1000 --treatment all
```

Report modelli:

```bash
uv run python scripts/build_model_report.py
```

Avvio locale:

```bash
briscola-server --reload
```
