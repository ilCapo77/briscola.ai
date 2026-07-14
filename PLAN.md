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
- La pista belief v1 multi-stile è chiusa con **STOP runtime**. Sui sette holdout migliora BCE macro del `7,63%`
  relativo e top-k di `+3,29` punti, ma nel PIMC 16×8 perde `-0,224` punti/partita contro v0 (CI95
  `-0,572..+0,124`) e fallisce lo screen preregistrato. Nessuna conferma 10k: belief v0 resta ufficiale. Report:
  `docs/plans/belief-v1-multistile-2026-07-14.md`.
- La diagnostica passiva A2C su tre seed × 2.000 partite supera tutti i gate. Il critic, pur ripartendo da zero,
  raggiunge explained variance mediana finale `0,127..0,133`; bias advantage `0,147..0,174`, picco gradiente
  `p95/mediana 1,92..2,00` e passi relativi sotto `0,019%` non indicano instabilità. **STOP**, per ora, a critic
  reuse, normalizzazione e clipping. Report:
  `docs/plans/a2c-health-diagnostic-v0-2026-07-14.md`.
- La schedule A2C realmente paired e' implementata e validata, ma il confronto 3 seed e' **inconcludente**. A pari
  20k partite perde direttamente in due seed su tre (mediana `-0,151`) e la dispersione della forza e' `2,08x`
  quella seriale; a pari 20k mazzi richiede 40k partite e resta direttamente neutra (`-0,148` mediano). Nessun run
  piu' lungo: `serial` resta il default. Report: `docs/plans/a2c-paired-schedule-v0-2026-07-14.md`.
- L'audit policy regret 192x64 e' chiuso con `no_policy_error_signal`. Su 96 decisioni early/mid trova 36 alternative
  candidate ma **zero errori affidabili al 99%**; i 14 errori confermati sono tutti gia' nelle finestre PIMC (9) o
  solver (5). Nessun cluster autorizza training o architetture nuove. Report:
  `docs/plans/policy-regret-audit-v14-2026-07-14.md`.
- Il diario pubblico condensa la linea di simmetria nei capitoli 18-19; i quattro approfondimenti tecnici restano
  separati, con l'esito finale in `docs/diario/21-una-voce-sola.md`.
- La produzione ha attualmente l'override `BRISCOLA_DEBUG_STATE_ENDPOINT=unsafe-full-state`: il tasto `S` funziona,
  ma chi conosce un game id può leggere mani e prossima carta. Il codice resta chiuso per default; rimuovere l'override
  quando il debug pubblico non serve più.

## Prossima Decisione

Le sette piste e l'audit degli errori residui hanno ora un esito chiuso. Rami e motivazioni complete in
`docs/plans/prossima-iterazione-modello.md`.

1. **Congelare la ricerca modello.** V14, PIMC belief 16x8, solver e schedule seriale restano la baseline; nessun
   risultato corrente autorizza v15, una nuova architettura o un altro training lungo.
2. **Correggere lo split dei dataset prima di riusare BC/value.** Raggruppare train/validation/test per partita,
   impedendo che stati della stessa partita finiscano in split diversi; aggiungere test e metadati di provenance.
3. **Aggiornare periodicamente il replay live.** Le nuove partite v14 aumentano la confidenza comportamentale, ma il
   basso volume previsto non blocca la diagnostica e non diventa training senza una nuova decisione su privacy e qualità.

Critic reuse, normalizzazione, clipping, nuove architetture e Q Monte Carlo restano sospesi finché una misura non
mostra il problema specifico che dovrebbero risolvere.

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

Avvio locale:

```bash
briscola-server --reload
```
