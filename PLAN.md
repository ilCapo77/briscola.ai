# Piano operativo — Briscola AI

> Questo file è volutamente breve: fotografa lo stato reale e le prossime decisioni.
> Cronologia e numeri completi vivono in `docs/plans/`, nel
> [diario di bordo](https://ai.briscola.dev/diario) e in `docs/diario/`.

## Stato Corrente

- Release corrente del repository: `0.36.0`. Il push di `master` attiva automaticamente il deploy su
  <https://ai.briscola.dev>; dopo ogni release verificare `/version`, asset ausiliari ed event log Postgres.
- Il default effettivo della UI è `bc_model_pimc_belief_16x8` su `best_a2c_v14.npz`, senza guard runtime
  anti-overkill. Il probe v14 32×8 vs 16×8 fa solo `+0,298` punti (CI95 `-0,025..+0,621`) a costo search
  `1,995×`: **STOP dose, 64×8 e budget adattivo; confermato 16×8**. La 64×10 resta solo variante selezionabile.
- `best_a2c_v14.npz` usa encoder v4 (`feature_dim=369`, hidden 256) e un solo forward normale. È distillata dalla
  media esatta dei logits v13 sulle 24 rinomine dei semi, su 50.000 partite e 1,9 milioni di decisioni. Il flip
  dell'argmax scende da `18,19%` a **`6,04%`** e l'overkill sui piatti poveri da `8,01%` a **`4,17%`**.
- I gate di promozione v14 sono positivi: policy-only vs v13 **`+0,66`** punti/partita (CI `+0,24..+1,09`) e default
  PIMC belief 16×8 vs v13 **`+0,43`** (CI `+0,03..+0,84`) su 10.000 partite seat-fair. Il controllo omogeneo big
  100k Numba contro `heuristic_v1` fa `+21,76` (CI `+21,59..+21,93`) ed è l'ultima riga del grafico del report.
- I tentativi precedenti paired-RL, forward-KL e margin hinge restano chiusi: non hanno raggiunto insieme il gate di
  simmetria e quello di forza. Cronologia, criteri e artefatti sono in `docs/plans/suit-*.md`; il resoconto completo
  della policy promossa è `docs/plans/suit-distillation-v0-2026-07-11.md`.
- La pista capacità ReLU è chiusa. V14 ha 123/256 unità quasi inattive; l'ablation congiunta è neutra su 10.000
  partite. Reset 8/16 le riattiva davvero, ma reset 16 migliora la KL validation solo dello **`0,328%`** rispetto
  allo stesso training senza reset, sotto il gate dell'`1%`. **STOP widening, potatura e altri reset.** Report:
  `docs/plans/hidden-unit-diagnostic-v0-2026-07-12.md` e `docs/plans/dormant-reinitialization-screen-v0-2026-07-12.md`.
- Runtime web zero-Numba: dominio, search PIMC e solver usano Python nel processo web; Numba resta per training,
  valutazioni e benchmark. In produzione: FastAPI Cloud, Redis per stato/pub-sub, Postgres in modalità `dataset`.
- Il catalogo modelli espone v14 come policy compatibile e non espone value/belief (`value_mlp_v1`/`belief_mlp_v1`),
  che sono asset interni. Policy v10/v11/v13/v14, value e belief necessari al runtime sono tracciati in Git; gli altri
  `.npz`, dataset e benchmark restano locali e gitignored.
- L'event log live contiene `human_action`, `ai_action`, `game_finished`, consenso e metadati modello.
  `export_live_actions.py` produce la sequenza unica umano+IA e può filtrare versione, agente, modello e bot.
- Il primo replay appaiato di campo usa 70 partite complete (59 v13, 11 v14) e 2.660 decisioni non forzate. Sulle
  stesse osservazioni i runtime concordano nell'`89,40%` dei casi; non emerge un difetto v14 localizzato e v14 riduce
  ancora overkill (`22,91% -> 20,54%`) e sprechi (`4 -> 1`). Il dato live resta diagnostico e non bloccante:
  `docs/plans/live-policy-replay-v13-v14-2026-07-14.md`.
- Il gate belief v1 multi-stile è implementato: roster congelato di sette stili, split per partita,
  leave-one-opponent-out, BCE/top-k/Brier/ECE e stop automatico. Il pilot da 770 partite valida soltanto la pipeline;
  il prossimo job probatorio usa 66.000 partite. Protocollo: `docs/plans/belief-v1-multistile-2026-07-14.md`.
- Il diario pubblico condensa la linea di simmetria nei capitoli 18-19; i quattro approfondimenti tecnici restano
  separati, con l'esito finale in `docs/diario/21-una-voce-sola.md`.
- La produzione ha attualmente l'override `BRISCOLA_DEBUG_STATE_ENDPOINT=unsafe-full-state`: il tasto `S` funziona,
  ma chi conosce un game id può leggere mani e prossima carta. Il codice resta chiuso per default; rimuovere l'override
  quando il debug pubblico non serve più.

## Prossima Decisione

La promozione v14 chiude la pista di simmetria; diagnostica e training controllato chiudono la capacità dormiente;
storico e probe v14 chiudono anche la dose PIMC. Rami esclusi in `docs/plans/prossima-iterazione-modello.md`.

1. **Eseguire il gate belief v1 completo.** Il runner genera 66.000 partite multi-stile, misura sette fold esclusi e
   allena il candidato all-styles solo se supera tutti i gate offline preregistrati.
2. **Solo dopo un GO offline, confrontare belief v1-v0 nel PIMC 16×8.** Screening seat-fair da 2.000 partite e
   conferma da 10.000; policy, solver, dose e uniform mix restano identici.
3. **Aggiornare periodicamente il replay live.** Le nuove partite v14 aumentano la confidenza comportamentale, ma il
   basso volume previsto non blocca il gate belief e non diventa training senza una nuova decisione su privacy e qualità.

Modifiche strutturali più ampie restano successive al gate belief e all'audit di campo.

## Audit Di Campo

- Le piste carichi guidati, timing dell'asso di briscola e cavata con mano lunga sono chiuse dalle ablation
  controfattuali: non riaprirle sulla base di singoli aneddoti. Riferimenti:
  `docs/plans/audit-campo-2026-07-07.md` e capitolo 17.
- Il replay appaiato corrente non confronta i rapporti vittorie grezzi: mostra v13 e v14 sulle stesse osservazioni e
  separa fallback, search e solver. Aggiornarlo quando il volume v14 cresce, separando versione e bot/load test.
- I dati umani restano diagnostici: niente training prima di rivalutare volume, consenso, qualità e privacy.

## Vincoli Operativi

- Anti-cheat: agenti, reward e modelli ricevono solo `PlayerObservation`; la vista full-state è debug opt-in e deve
  restare `403` di default.
- Se cambia una regola o l'osservazione, mantenere allineati dominio canonico, fast path e Numba con test di parità.
- Valutazioni serie: seat-fair paired, stesso mazzo e posti scambiati, CI sulle coppie. Miglioramenti sotto il punto
  richiedono anche più seed di training: la CI di evaluation non misura la varianza del training.
- Confrontare varianti search a pari CPU media e p95, non soltanto a pari numero di determinizzazioni.
- I job oltre ~5 minuti vanno preparati per il maintainer con `nohup`, log e path artefatti; non vanno avviati
  dall'agente restando in attesa.
- Ogni bump richiede `pyproject.toml` + `uv.lock`, tag annotato e rigenerazione/verifica di
  `docs/reports/model_progress.xlsx`; il catalogo deve mostrare la policy compatibile e nascondere value/belief.

## Debito Aperto

- BC e value usano ancora split casuali per record: prima di riutilizzarli per una nuova linea, introdurre split per
  partita per evitare leakage tra stati della stessa partita.
- Il trainer A2C va diagnosticato prima di aggiungere PPO: salute/gradienti del trunk, critic reinizializzato nei
  warm-start, normalizzazione advantage e gradient clipping sono ablation separate, non un unico cambio.
- RNG seriale e parallelo di `decision_quality` non è riproducibile cross-`workers`.
- Cold start residuo (~13.7s) è il pavimento della piattaforma; leve reali: keep-alive sotto l'idle timeout o piano
  diverso. Il runtime applicativo non ha più compilazione JIT.

## Comandi Utili

Quality gate:

```bash
uv run ruff format src tests scripts
uv run ruff check --fix src tests scripts
uv run mypy src
make docs-check
uv run pytest
```

Verifica deploy e catalogo:

```bash
curl -sS https://ai.briscola.dev/version
curl -sS https://ai.briscola.dev/api/ai/models
```

Export live v14 per il prossimo audit:

```bash
DATABASE_URL=... uv run python scripts/export_live_actions.py \
  --code-version 0.36.0 \
  --ai-agent bc_model_pimc_belief_16x8 \
  --ai-model-id best_a2c_v14.npz \
  --exclude-client-id loadtest-bot \
  --out data/prod_live_actions_v14.jsonl
```

Gate locale e report:

```bash
uv run python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model --agent0-model data/models/best_a2c_v14.npz \
  --agent1 bc_model --agent1-model data/models/best_a2c_v13.npz
uv run python scripts/build_model_report.py
```

Sonda riproducibile di simmetria dei semi:

```bash
uv run python scripts/probe_suit_symmetry.py \
  --model data/models/best_a2c_v14.npz \
  --out-json data/suit_symmetry_v14.json
```

Gate belief v1 completo (job lungo, riprendibile):

```bash
nohup caffeinate -i uv run python scripts/run_belief_v1_gate.py \
  --resume \
  > data/belief/belief_v1_gate_20260714.log 2>&1 &
```

Avvio locale:

```bash
briscola-server --reload
```
