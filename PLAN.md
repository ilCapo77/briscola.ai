# Piano operativo — Briscola AI

Questo file è la fonte di verità per decidere cosa fare dopo. Deve restare breve: i dettagli storici completi vivono
nei commit, nei test e nei report.

## Stato Corrente

- Versione progetto: `0.19.0`.
- Produzione: <https://briscolaai.fastapicloud.dev>.
- Modello consigliato: `best_a2c_v7.npz` (encoder v3, `feature_dim=310`, guard anti-overkill ON).
- Default UI: `bc_model` + modello consigliato, cioè v7 puro. È la nuova policy `.npz` veloce promossa in v0.19.0.
- Seconda scelta vicina nel menu: `bc_model_value_lookahead_8x8`, cioè modello selezionato (default v7) + solver
  finale + V-lookahead depth-1 quando restano al massimo 8 carte vive ignote. Resta l'opzione runtime più forte,
  ma costa più CPU.
- Altro avversario avanzato selezionabile: `bc_model_pimc_16x8`, cioè PIMC(v6, 16 determinizzazioni, max 8 carte
  vive ignote) + solver finale.
- Backend: FastAPI + WebSocket, stato in `GameSessionStore` (Redis in cloud), event log SQLite/Postgres.
- Dataset cloud: `DATABASE_URL` + `BRISCOLA_EVENT_LOG_MODE=dataset`; il backend salva eventi `ai_action` auditabili
  per mosse IA/search (`fallback`, `lookahead`, `search`, `solver`).
- Diagnostica cloud: `/version` e `/api/meta` espongono `event_log_available`, `event_log_healthy`,
  `event_log_backend`, `event_log_database_name` ed `event_log_database_host` per verificare che il processo live
  abbia un event log Postgres raggiungibile e collegato al Neon atteso.
- Debug UI: la vista full-state di `GET /api/games/{id}` (senza `player_index`, include `next_deck_card`) è ora
  **opt-in** via `BRISCOLA_DEBUG_STATE_ENDPOINT=1`; di default risponde 403 (anti-cheat in produzione). La vista
  fair (`player_index`) e gli agenti continuano a ricevere solo `ObservationDTO`.
- Anti-cheat: agenti e modelli ricevono solo `PlayerObservation`, mai `GameState` completo.
- CI: GitHub Actions (`.github/workflows/ci.yml`) esegue ruff format/check, mypy e pytest+coverage su ogni push/PR.
- Test-àncora anti-divergenza: `tests/test_card_tables_parity.py` (tabelle punti/forza e `who_wins_trick` di fast,
  numba core, numba solver ed encoding contro `domain.models.Rank`; parità `_card_to_id_fast`↔`card_to_id`) e
  `tests/test_reward_shaping_numba_parity.py` (overkill penalty JIT ↔ reward shaping canonico); il test del solver
  endgame Numba verifica anche `final_delta_p0_p1` (valore di foglia del value-lookahead).
- Artefatti locali (`data/`, `benchmarks/`) restano gitignored.

## Decisioni Chiuse

### v7 È Il Default `.npz`

`best_a2c_v7.npz` è il modello consigliato ufficiale.

Numeri di promozione principali:

- v7 vs v6 big holdout seat-fair: `+2.27` punti medi su 100k partite, CI95 `+2.11..+2.44`.
- `v7 + solver` vs `v6 + solver`, 10k: `+2.27`, CI95 `+1.77..+2.77`.
- v7 vs `heuristic_v1` big holdout: `+18.73`.
- decision-quality medium vs `heuristic_v1`: `trump_overkill_rate=0.0%`; `trump_waste_rate` circa `0.3%`
  contro circa `0.1%` di v6 nello stesso harness, da monitorare.

Contesto: v7 non supera il value-lookahead runtime basato su v6; nel confronto 10k `v7 + solver` perde di `-0.64`,
CI95 `-1.15..-0.13`. Quindi v7 diventa il default veloce `.npz`, mentre value-lookahead resta l'opzione avanzata più
forte. Smoke 2k: `value-lookahead(v7)` batte `value-lookahead(v6)` di `+1.93`, da confermare solo se serve.

### Solver Endgame È Deployabile

Il solver endgame 2-player a mazzo vuoto è esatto. `ai/endgame/solver.py` resta l'oracolo didattico basato su
`domain.step()`, `ai/endgame/fast_solver.py` è il solver completo numerico/Python, e il loop caldo usa
`ai/endgame/numba_solver.py` choose-only JIT, equivalente e coperto da test di parità. Viene eseguito solo dopo
ricostruzione da `PlayerObservation`.

Decisione: il solver resta una variante runtime valida e a basso rischio; ora può essere usato sopra v7 per confronti
offline, mentre il default UI resta `bc_model` puro con `best_a2c_v7.npz`.

### PIMC È Utile Come Runtime, Non Come Modello Distillato

Evidenza offline:

- `PIMC(v6,16×10)` vs `v6 + solver`, 2000 partite: avg diff `+3.59`, CI95 `+2.48..+4.70`.
- `PIMC(v6,16×8)` vs `16×10`, 2000 partite: nessuna evidenza che `16×10` sia più forte; `16×8` costa meno.
- allargare la finestra a `16×12`/`16×16` aumenta CPU/search move ma non mostra vantaggio.

Decisione: mantenere `bc_model_pimc_16x8` come avversario avanzato selezionabile, non come default.

Caveat: i benchmark misurano forza contro v6 nell'harness offline. Contro umani il modello d'avversario nei rollout è
mis-specificato; validare con audit qualitativo reale.

### Distillazione PIMC In Policy: Negativa Per Ora

Esperimenti chiusi:

- hard-label PIMC su correzioni search: val acc circa `57%`, non abbastanza.
- mix deploy-relevant `(coverage v6 + correzioni PIMC pesate)`: peggiora `(candidato+solver)` vs `(v6+solver)`.
- MLP più larga (`hidden=512`) memorizza il train ma non migliora la generalizzazione.
- soft-label da `mean_score` PIMC: T=`2` best val acc `56.9%`, T=`5` `55.6%`, T=`10` `52.8%`; non supera hard-label.

Decisione: non continuare distillazione PIMC in questa MLP/recipe senza una nuova ipotesi sostanziale.

### V-Lookahead Stage 0: Positivo

Ipotesi: non distillare l'argmax PIMC in una policy reattiva; allenare invece un value model `V(observation)` e usarlo
per ordinare foglie di una lookahead corta.

Stage 0 validato con dataset `v6 + solver`, `epsilon=0.10`, label pulita `v6-continuation`:

- 50k partite, 2M record value-observation.
- `train_value.py`: MAE `14.02` vs baseline delta corrente `16.34`; sign acc `0.734` vs `0.633`; best checkpoint epoca
  16.
- gate ranking vs diagnostica PIMC 64×8, 5000 record search: top1 `0.7276` vs reference v6 `0.5278`;
  strong/reliable top1 `0.8395` vs `0.6359`; pairwise `0.8016`, strong pairwise `0.8502`; `records_failed=0`.

Decisione: esporre `modello selezionato + solver + V-lookahead depth-1` come avversario selezionabile vicino a
`bc_model`. Resta più forte del solo `.npz` v7, ma costa più CPU e non diventa default.

### A2C Contro Value-Lookahead: Chiuso Positivo

`train_a2c.py` può usare `bc_model_value_lookahead_8x8` come opponent nel fast rollout Numba. In questo path
l'avversario usa la partita numerica determinizzata come singola determinizzazione (`fast_numba_determinized`):
è un opponent forte per training/screening, non una replica bit-a-bit dell'agente UI.

Run effettivo:

- 5M partite, warm-start v6, opponent `bc_model_value_lookahead_8x8`, seed `20260701`.
- esito: promosso come `best_a2c_v7.npz`.

Decisione:

- non ripetere subito lo stesso training;
- usare v7 come nuova policy base per confronti futuri;
- se si vuole migliorare ancora, partire da audit reali o da un nuovo value model/value-lookahead con v7 come base.

### Population League Declassata

Il round-robin `{best_a2c_v2, v3, v4, v5, v6, heuristic_v1}` mostra famiglia monotona/transitiva: nessun ciclo
confidente.

Decisione: niente population league come asse primario. Si può fare solo uno screening economico di robustezza, con
kill criterion esplicito, se nasce un motivo nuovo.

## Prossime Azioni

### 1. Value Model Decision-Aligned Chiuso

Stato: due varianti di riaddestramento del value model sono state misurate e non battono `value_v0`.

- `value_v1_h128_v7_window500k_seed20260701.npz` migliora molto MAE/sign sul proprio holdout, ma non migliora
  materialmente il value-lookahead runtime;
- ranking gate vs diagnostica PIMC: top1 circa pari a `value_v0`;
- A/B runtime diretto `value_v1` vs `value_v0`, 4k seat-fair: `+0.24`, CI95 `-0.55..+1.04`;
- decisione: non promuovere `value_v1`; tenere `value_v0` come value model deployato.

Ipotesi successiva testata: `V` deve imparare a ordinare le foglie come la search PIMC, non solo predire l'esito della
continuazione base.

Implementazione pronta:

- `scripts/generate_value_dataset_numba.py` genera dataset `.npz` compatti da self-play numerico JIT;
- `scripts/generate_pimc_leaf_value_dataset.py` converte diagnostica PIMC root-level in foglie value decision-aligned;
- `train_value.py` legge sia JSONL canonico sia `.npz` compatto.
- `scripts/train_value_pairwise.py` allena `V` con regressione + ranking pairwise intra-root, con warm-start opzionale da
  `value_v0`;
- `scripts/evaluate_value_lookahead_pair.py` confronta due value model nello stesso harness value-lookahead.

Probe tecnico con teacher PIMC v6:

- usando diagnostica PIMC v6 esistente e policy base v7, dataset 5k root / 15k leaf;
- training da zero perde nettamente contro `value_v0` (`-2.05`, CI95 `-3.62..-0.47`, 1k);
- fine-tune da `value_v0` è sano offline ma non supera `value_v0` (`-0.55`, CI95 `-2.14..+1.03`, 1k);
- lettura: il plumbing è valido, ma il teacher v6 è disallineato. Non usare questo artefatto per promozione.

Probe definitivo con teacher PIMC v7:

- teacher `best_a2c_v7.npz`, `d=64`, `u=8`, 40k record:
  - `records_written_search=18184`, `records_written_endgame_solver=21816`;
  - `records_written_search_disagree_reference=8176`;
  - `records_written_search_strong_reliable_disagree=2935`;
  - `teacher_seconds_per_search_decision=0.0718`;
- leaf dataset decision-aligned:
  - `roots_used=5022`, `leaf_records_written=15066`;
  - `leaf_records_skipped_terminal_or_endgame=12684`;
  - `leaf_records_skipped_error=0`;
- fine-tune da `value_v0`:
  - best epoch 15;
  - validation pairwise `0.809`, top1 `0.735`;
- A/B runtime diretto, 4k seat-fair, seed `20260710`:
  - `value_leaf_pairwise_v7_40k` vs `value_v0`;
  - score rate `0.4888`, CI95 `0.4733..0.5042`;
  - avg diff `-0.69`, CI95 `-1.48..+0.11`.

Decisione:

- non promuovere `value_leaf_pairwise_v7_40k`;
- mantenere `value_v0_h128_clean50k_seed20260701.npz` come value model dell'agente `bc_model_value_lookahead_8x8`;
- non rilanciare altri fine-tune value piccoli senza una nuova ipotesi strutturale;
- se si riprende questo asse, servono o una diversa architettura/capacità per `V`, o un test mirato su errori reali
  raccolti da audit, non un altro dataset simile.

### 2. Monitoraggio Produzione E Audit Value-Lookahead/PIMC

Fare:

- mantenere `bc_model_value_lookahead_8x8` come opzione avanzata vicina al default;
- mantenere `bc_model_pimc_16x8` come confronto avanzato opzionale;
- classificare ogni mossa sospetta dal `decision_type`: se è `fallback`, il problema è la policy base; se è `solver`,
  verificare ricostruzione/endgame; se è `lookahead`/`search`, verificare value/PIMC/guard/determinizzazione;
- se emergono errori ricorrenti, trasformarli in una nuova ipotesi misurabile.

Non fare:

- non usare dati umani per training finché volume, consenso, qualità e privacy non sono riverificati;
- non avviare v8 per inerzia.

### 3. Hardening Continuo

Già aggiunti stress test su:

- solver reale vs solver ricostruito da `PlayerObservation`, anche secondo di mano;
- determinizzazioni PIMC senza duplicati/leak e con punteggi pubblici coerenti;
- PIMC search senza determinizzazioni/rollout falliti né mosse normalizzate;
- solver/lookahead Numba usabili nei loop di training.

Continuare ad aggiungere test solo quando troviamo un caso reale sospetto o tocchiamo regole/observation/PIMC.

### 4. Nuovo Modello Solo Con Nuova Ipotesi

Un nuovo `best_a2c_v8` ha senso solo se c'è un segnale concreto, ad esempio:

- pattern di errore V-lookahead/PIMC/v7 ripetibile da dataset reale;
- nuova architettura/feature che risolve un limite osservato;
- nuovo value model/value-lookahead con v7 come base e segnale preliminare forte, misurato prima come agente runtime;
- aumento di volume umano sufficiente e privacy/qualità verificata.

Qualunque promozione deve includere:

- confronto seat-fair con CI;
- holdout non peggiore;
- decision-quality vs `heuristic_v1`;
- `trump_waste_rate` e `trump_overkill_rate` non peggiorati materialmente;
- aggiornamento `docs/reports/model_progress.xlsx` se cambia un best ufficiale.

### 5. PPO/GAE: Bassa Priorità

Valutare PPO/GAE solo dopo un blocco reale di A2C/PIMC e con esperimento piccolo e isolato. Non introdurre DQN per ora:
action mask, osservabilità parziale e self-play rendono più coerente restare su policy-gradient.

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

Value-learning / V-lookahead:

```bash
uv run python scripts/generate_value_dataset.py --help
uv run python scripts/generate_value_dataset_numba.py --help
uv run python scripts/train_value.py --help
uv run python scripts/evaluate_value_ranking.py --help
uv run python scripts/evaluate_value_lookahead.py --help
uv run python scripts/evaluate_value_lookahead_quality.py --help
```

Avvio locale:

```bash
briscola-server --reload
```
