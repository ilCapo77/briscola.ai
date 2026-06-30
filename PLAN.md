# Piano operativo — Briscola AI

Questo file è la fonte di verità per decidere cosa fare dopo. Deve restare breve: i dettagli storici completi vivono
nei commit, nei test e nei report.

## Stato Corrente

- Versione progetto: `0.15.0`.
- Produzione: <https://briscolaai.fastapicloud.dev>.
- Modello consigliato: `best_a2c_v6.npz` (encoder v3, `feature_dim=310`, guard anti-overkill ON).
- Default UI: `bc_model_hybrid_endgame` + modello consigliato, cioè v6 durante la partita e solver esatto a mazzo
  vuoto.
- Avversario avanzato selezionabile: `bc_model_pimc_16x8`, cioè PIMC(v6, 16 determinizzazioni, max 8 carte vive
  ignote) + solver finale.
- Backend: FastAPI + WebSocket, stato in `GameSessionStore` (Redis in cloud), event log SQLite/Postgres.
- Dataset cloud: `DATABASE_URL` + `BRISCOLA_EVENT_LOG_MODE=dataset`; in `v0.15.0` il backend salva anche eventi
  `ai_action` auditabili per mosse IA/PIMC.
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

Decisione: `v6 + solver` è il default prudente in produzione.

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

Decisione: procedere a Stage 1, cioè agente `v6 + solver + V-lookahead depth-1`, e misurarlo seat-fair contro
`v6 + solver`. Non è ancora un nuovo best né un default UI.

### Population League Declassata

Il round-robin `{best_a2c_v2, v3, v4, v5, v6, heuristic_v1}` mostra famiglia monotona/transitiva: nessun ciclo
confidente.

Decisione: niente population league come asse primario. Si può fare solo uno screening economico di robustezza, con
kill criterion esplicito, se nasce un motivo nuovo.

## Prossime Azioni

### 1. Stage 1 V-Lookahead

Obiettivo: verificare se il value model validato offline produce forza reale quando usato a runtime.

Fare:

- implementare agente domain-only `V-lookahead depth-1 + solver`: per ogni carta legale applica la mossa, risolve la
  presa corrente con policy `v6 + solver`, valuta la foglia con `V`, cambia segno se la foglia tocca all'avversario;
- fallback a `v6 + solver` in caso di errore, e solver esatto a mazzo vuoto;
- testare anti-cheat, determinismo seed e parità del caso `depth=0`/fallback;
- valutare seat-fair contro `v6 + solver`, 2000-4000 partite, con CI su score e avg diff;
- promuovere solo se il lower bound CI95 di avg diff è materialmente positivo e la latenza/mossa è compatibile col
  cloud.

Non fare:

- non esporlo in UI prima del confronto head-to-head;
- non chiamarlo v7 finché non batte `v6 + solver` in evaluation.

### 2. Monitoraggio Produzione E Audit PIMC

Obiettivo: capire come si comportano `v6 + solver` e `PIMC(v6,16×8)` contro giocatori reali, senza usare ancora questi
dati per training.

Fare:

- far giocare 10-20 partite umane contro `bc_model_pimc_16x8`;
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

- pattern di errore PIMC/v6 ripetibile da dataset reale;
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
```

Avvio locale:

```bash
briscola-server --reload
```
