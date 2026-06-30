# Piano operativo — Briscola AI

Questo file è la fonte di verità per decidere cosa fare dopo. Deve restare breve: i dettagli storici completi vivono
nei commit, nei test e nei report.

## Stato Corrente

- Versione progetto: `0.17.0`.
- Produzione: <https://briscolaai.fastapicloud.dev>.
- Modello consigliato: `best_a2c_v6.npz` (encoder v3, `feature_dim=310`, guard anti-overkill ON).
- Default UI: `bc_model` + modello consigliato, cioè v6 puro. È la baseline più leggibile per giocatori umani e audit.
- Seconda scelta vicina nel menu: `bc_model_value_lookahead_8x8`, cioè v6 + solver finale + V-lookahead depth-1
  quando restano al massimo 8 carte vive ignote.
- Altro avversario avanzato selezionabile: `bc_model_pimc_16x8`, cioè PIMC(v6, 16 determinizzazioni, max 8 carte
  vive ignote) + solver finale.
- Backend: FastAPI + WebSocket, stato in `GameSessionStore` (Redis in cloud), event log SQLite/Postgres.
- Dataset cloud: `DATABASE_URL` + `BRISCOLA_EVENT_LOG_MODE=dataset`; il backend salva eventi `ai_action` auditabili
  per mosse IA/search (`fallback`, `lookahead`, `search`, `solver`).
- Diagnostica cloud: `/version` e `/api/meta` espongono `event_log_available`, `event_log_backend` e
  `event_log_database_name` per verificare che il processo live stia davvero scrivendo su Postgres.
- Anti-cheat: agenti e modelli ricevono solo `PlayerObservation`, mai `GameState` completo.
- Artefatti locali (`data/`, `benchmarks/`) restano gitignored.

## Decisioni Chiuse

### v6 È La Baseline

`best_a2c_v6.npz` resta il modello ufficiale.

Numeri di promozione principali:

- v6 vs v5 big guarded seat-fair: `+0.46` punti medi su 100k partite.
- holdout v6 vs v5 big: `+0.46`.
- v6 vs `heuristic_v1`: `+18.40`.
- decision-quality vs `heuristic_v1`: `+18.58`, `trump_overkill_rate=0.0%`, `trump_waste_rate=0.07%`.

Non fare un v7 solo replicando lo stesso recipe con più partite: lo scaling policy-only ha rendimento ormai piccolo.

### Solver Endgame È Deployabile

Il solver endgame 2-player a mazzo vuoto è esatto, usa `domain.step()` e viene eseguito solo dopo ricostruzione da
`PlayerObservation`.

Decisione: `v6 + solver` resta una variante runtime valida e a basso rischio, ma non è il default UI mentre raccogliamo
feedback umano comparabile sul v6 puro e sugli agenti avanzati.

### PIMC È Utile Come Runtime, Non Come Modello Distillato

Evidenza offline:

- `PIMC(v6,16×10)` vs `v6 + solver`, 2000 partite: avg diff `+3.59`, CI95 `+2.48..+4.70`.
- `PIMC(v6,16×8)` vs `16×10`, 2000 partite: nessuna evidenza che `16×10` sia più forte; `16×8` costa meno.
- allargare la finestra a `16×12`/`16×16` aumenta CPU/search move ma non mostra vantaggio.

Decisione: mantenere `bc_model_pimc_16x8` come avversario avanzato selezionabile, non come default.

Caveat: i benchmark misurano forza contro v6 nell'harness offline. Contro umani il modello d'avversario nei rollout è
mis-specificato; validare con audit qualitativo reale.

### Distillazione PIMC In v7: Negativa Per Ora

Esperimenti chiusi:

- hard-label PIMC su correzioni search: val acc circa `57%`, non abbastanza.
- mix deploy-relevant `(coverage v6 + correzioni PIMC pesate)`: peggiora `(v7+solver)` vs `(v6+solver)`.
- MLP più larga (`hidden=512`) memorizza il train ma non migliora la generalizzazione.
- soft-label da `mean_score` PIMC: T=`2` best val acc `56.9%`, T=`5` `55.6%`, T=`10` `52.8%`; non supera hard-label.

Decisione: non promuovere nessun `pimc_v7_*` e non continuare distillazione PIMC in questa MLP/recipe senza una nuova
ipotesi sostanziale.

### V-Lookahead Stage 0: Positivo

Ipotesi: non distillare l'argmax PIMC in una policy reattiva; allenare invece un value model `V(observation)` e usarlo
per ordinare foglie di una lookahead corta.

Stage 0 validato con dataset `v6 + solver`, `epsilon=0.10`, label pulita `v6-continuation`:

- 50k partite, 2M record value-observation.
- `train_value.py`: MAE `14.02` vs baseline delta corrente `16.34`; sign acc `0.734` vs `0.633`; best checkpoint epoca
  16.
- gate ranking vs diagnostica PIMC 64×8, 5000 record search: top1 `0.7276` vs reference v6 `0.5278`;
  strong/reliable top1 `0.8395` vs `0.6359`; pairwise `0.8016`, strong pairwise `0.8502`; `records_failed=0`.

Decisione: esporre `v6 + solver + V-lookahead depth-1` come avversario selezionabile vicino a `bc_model`. Non è ancora
un nuovo best `.npz` né il default UI.

### Population League Declassata

Il round-robin `{best_a2c_v2, v3, v4, v5, v6, heuristic_v1}` mostra famiglia monotona/transitiva: nessun ciclo
confidente.

Decisione: niente population league come asse primario. Si può fare solo uno screening economico di robustezza, con
kill criterion esplicito, se nasce un motivo nuovo.

## Prossime Azioni

### 1. Monitoraggio Produzione Value-Lookahead

Obiettivo: osservare `bc_model_value_lookahead_8x8` in produzione contro giocatori reali prima di qualunque altra
promozione o training.

Stato:

- implementati `ValueLookaheadAgent` e `scripts/evaluate_value_lookahead.py`;
- deployato in `v0.16.0` come seconda scelta vicina a `bc_model`; default UI resta v6 puro;
- `v0.17.0` aggiunge diagnostica event-log runtime e `scripts/inspect_event_log_game.py` per isolare mismatch DB/log;
- provisioning cloud attivo per `value_v0_h128_clean50k_seed20260701.npz` via
  `BRISCOLA_VALUE_MODEL_URL`/`BRISCOLA_VALUE_MODEL_SHA256`;
- il guard anti-overkill è attivo di default sulle sole decisioni V-lookahead; fallback e solver restano invariati;
- held-out 4000 partite vs `v6 + solver`, seed diverso: avg diff `+2.65`, CI95 `+1.85..+3.45`; score rate `0.5421`,
  CI95 `0.5267..0.5575`; `0` determinizzazioni/leaf eval fallite; circa `0.016s` per mossa lookahead;
- decision-quality medium vs `heuristic_v1`: avg diff `+20.09` vs baseline `v6+solver` `+18.60`; trump waste
  `0.24%` vs `0.21%`; trump overkill `11.84%` vs `11.66%`; low-lead overkill `0.22%` vs `0.40%`.

Fare:

- far giocare 10-20 partite umane contro `bc_model_value_lookahead_8x8`;
- dopo ogni deploy verificare che `/version` mostri `event_log_mode=dataset`, `event_log_available=true`,
  `event_log_backend=postgres`, `event_log_database_name=neondb` (o il nome DB atteso);
- esportare/auditare gli eventi `ai_action`;
- verificare per ogni partita: conteggio `lookahead`/`solver`/`fallback`, `failed_determinizations=0`,
  `failed_leaf_evaluations=0`, `overkill_guard_adjustments`, e mosse qualitativamente sospette;
- classificare ogni mossa sospetta dal `decision_type`: se è `fallback`, il problema è v6; se è `solver`, verificare
  ricostruzione/endgame; se è `lookahead`, verificare value/guard/determinizzazione.

Non fare:

- non esporlo come default UI prima di una prova cloud/umana;
- non chiamarlo v7: non è un nuovo `.npz` policy, è un agente runtime;
- non avviare nuovo training finché non emerge un pattern reale misurabile dalle partite/audit.

### 2. Monitoraggio Produzione E Audit PIMC

Obiettivo: mantenere `PIMC(v6,16×8)` come confronto avanzato opzionale, senza spostare il focus dal V-lookahead appena
deployato.

Fare:

- far giocare qualche partita umana contro `bc_model_pimc_16x8` solo se serve un confronto qualitativo con
  V-lookahead;
- esportare/auditare le mosse IA con:
  - `scripts/audit_event_log_games.py`
  - `scripts/export_ai_actions.py`
  - `scripts/report_event_log.py`
- classificare ogni mossa sospetta come `fallback`, `search` o `endgame_solver`;
- se emergono errori ricorrenti, trasformarli in una nuova ipotesi misurabile.

Non fare:

- non usare dati umani per training finché volume, consenso, qualità e privacy non sono riverificati;
- non avviare v7 per inerzia.

### 3. Hardening Continuo

Già aggiunti stress test su:

- solver reale vs solver ricostruito da `PlayerObservation`, anche secondo di mano;
- determinizzazioni PIMC senza duplicati/leak e con punteggi pubblici coerenti;
- PIMC search senza determinizzazioni/rollout falliti né mosse normalizzate.

Continuare ad aggiungere test solo quando troviamo un caso reale sospetto o tocchiamo regole/observation/PIMC.

### 4. Nuovo Modello Solo Con Nuova Ipotesi

Un nuovo `best_a2c_v7` ha senso solo se c'è un segnale concreto, ad esempio:

- pattern di errore V-lookahead/PIMC/v6 ripetibile da dataset reale;
- nuova architettura/feature che risolve un limite osservato;
- teacher/search diverso con evidenza preliminare forte;
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
uv run python scripts/train_value.py --help
uv run python scripts/evaluate_value_ranking.py --help
uv run python scripts/evaluate_value_lookahead.py --help
uv run python scripts/evaluate_value_lookahead_quality.py --help
```

Avvio locale:

```bash
briscola-server --reload
```
