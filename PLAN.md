# Piano operativo — Briscola AI

> Questo file è volutamente breve: fotografa lo stato reale e le prossime decisioni.
> Cronologia e numeri completi vivono in `docs/plans/`, nel
> [diario di bordo](https://ai.briscola.dev/diario) e in `docs/diario/`.

## Stato Corrente

- Release corrente del repository: `0.35.1`. Il push di `master` attiva automaticamente il deploy su
  <https://ai.briscola.dev>; dopo ogni release verificare `/version`, asset ausiliari ed event log Postgres.
- Il default effettivo della UI è `bc_model_pimc_belief_16x8` su `best_a2c_v13.npz`, senza guard runtime
  anti-overkill. La 64×10 resta selezionabile come variante a budget massimo: nei gate storici a parità di policy era
  più forte ma circa 6× più costosa; il confronto non è stato ripetuto sulla v13.
- `best_a2c_v13.npz` usa encoder v4 (`feature_dim=369`, hidden 256). Rispetto a v11 è neutra sulla forza:
  policy-only `-0.03` (CI `-0.38..+0.32`) e default PIMC 16×8 `+0.14` (CI `-0.20..+0.47`). Riduce però l'overkill di
  briscola su piatti poveri da circa `28-31%` a `6-8%`: **stessa forza, comportamento migliore**.
- La sonda canonica di simmetria dei semi su 4.096 osservazioni v13 trova un flip dell'argmax nel **18,19%** delle
  94.208 rinomine non banali. L'ablation paired v0 su tre seed da 10k non supera il gate: flip medio `18,32% ->
  18,84%` e head-to-head paired-vs-control `-0,15` punti/partita. Il flag resta sperimentale e spento; niente run
  lunga. Metodo, numeri e interpretazione: `docs/plans/suit-augmentation-paired-v0-2026-07-11.md`.
- La consistency loss separata supera invece lo screening a tre seed: con beta `0.1` il flip medio scende a
  **15,64%** e la JS da `0,14124` a `0,10402` bit. È neutra contro v13 (`-0,08` punti/partita medi; tutte le CI
  includono zero). Il follow-up chiude però il ramo: a 30k il flip si ferma a `15,46%` con forza neutra (`-0,02`),
  a 50k risale a `16,47%` e la forza cala **`-0,77`** punti/partita, significativa su tutti i seed. La JS continua a
  scendere, ma il margine top-2 si comprime. **STOP forward-KL v0; nessun gate PIMC e nessuna promozione.** Dettagli:
  `docs/plans/suit-consistency-v0-2026-07-11.md`.
- La successiva hinge sul margine evita il collasso di sicurezza ma non raggiunge il gate: il migliore beta `0.3`
  porta il flip a **14,42%**, conserva il gap top-2 (`0,915` vs `0,929` dei controlli) ed è neutro contro v13
  (`-0,14` punti/partita). Beta `1.0` non migliora: la curva è satura. **STOP margin v0; niente run lungo.** Report:
  `docs/plans/suit-margin-v0-2026-07-11.md`.
- La media esatta dei logits v13 sulle 24 rinomine chiude la domanda causale: flip `0`, costo inference solo `1,45x`
  grazie al batch e vantaggio diretto **`+0,90` punti/partita** su v13 (CI `+0,47..+1,33`). Migliora anche sulle
  due baseline e dimezza l'overkill su piatti poveri (`8,0% -> 3,9%`). **GO alla distillazione; nessuna promozione
  runtime/PIMC del wrapper 24x.** Report: `docs/plans/suit-symmetrized-v13-2026-07-11.md`.
- Runtime web zero-Numba: dominio, search PIMC e solver usano Python nel processo web; Numba resta per training,
  valutazioni e benchmark. In produzione: FastAPI Cloud, Redis per stato/pub-sub, Postgres in modalità `dataset`.
- Il catalogo modelli espone v13 come policy compatibile e non espone value/belief (`value_mlp_v1`/`belief_mlp_v1`),
  che sono asset interni. Policy v10/v11/v13, value e belief necessari al runtime sono tracciati in Git; gli altri
  `.npz`, dataset e benchmark restano locali e gitignored.
- L'event log live contiene `human_action`, `ai_action`, `game_finished`, consenso e metadati modello.
  `export_live_actions.py` produce la sequenza unica umano+IA e può filtrare versione, agente, modello e bot.
- Il diario pubblico e gli approfondimenti tecnici arrivano al capitolo 20,
  `docs/diario/20-ventiquattro-pareri.md`.

## Prossima Decisione

La policy esattamente simmetrica gioca meglio: la domanda causale è chiusa. Ora bisogna trasferire il beneficio in un
modello singolo, senza introdurre il batch 24x nel default PIMC. Ipotesi e criteri delle altre piste restano in
`docs/plans/prossima-iterazione-modello.md`.

1. **Distillare il teacher simmetrizzato.** Costruire un corpus per partita, produrre target soft dalla media 24x e
   allenare una singola MLP v4. Prima verificare imitation su holdout, poi flip, direct match contro v13 e teacher,
   baseline e qualità decisionale. Se la MLP ordinaria perde il vantaggio o torna asimmetrica, passare a pesi
   condivisi fra semi; non riaprire un'altra variante della consistency loss.
2. **Completare la diagnostica delle ReLU.** Misurare activation rate, contributo ai logits e flip per ablation del
   neurone. Il widening `256→320/384` resta subordinato a un collo di bottiglia misurato; la simmetria trovata non è
   di per sé una prova che serva più capacità.
3. **Isolare la dose PIMC.** Confrontare 16/32/64 determinizzazioni con finestra 8, stessi seed e CPU media/p95. Se la
   curva è positiva, provare budget adattivo; se è piatta, chiudere la pista senza riaprire 8→10, già negativa.

Belief v1 e modifiche strutturali più ampie restano successive: una nuova belief va allenata contro un roster misto
v13/anchor/euristiche, validata leave-one-opponent-out e poi confrontata v0-vs-v1 nello stesso PIMC 16×8. Le metriche
offline da sole non bastano per la promozione.

## Audit Di Campo

- Le piste carichi guidati, timing dell'asso di briscola e cavata con mano lunga sono chiuse dalle ablation
  controfattuali: non riaprirle sulla base di singoli aneddoti. Riferimenti:
  `docs/plans/audit-campo-2026-07-07.md` e capitolo 17.
- Ripetere l'audit solo con qualche centinaio di partite umane complete contro v13, separando per data/versione e
  filtrando bot/load test. I dati umani restano diagnostici: niente training prima di rivalutare volume, consenso,
  qualità e privacy.

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
uv run pytest
```

Verifica deploy e catalogo:

```bash
curl -sS https://ai.briscola.dev/version
curl -sS https://ai.briscola.dev/api/ai/models
```

Export live v13 per il prossimo audit:

```bash
DATABASE_URL=... uv run python scripts/export_live_actions.py \
  --code-version 0.35.1 \
  --ai-agent bc_model_pimc_belief_16x8 \
  --ai-model-id best_a2c_v13.npz \
  --exclude-client-id loadtest-bot \
  --out data/prod_live_actions_v13.jsonl
```

Gate locale e report:

```bash
uv run python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model --agent0-model data/models/best_a2c_v13.npz \
  --agent1 bc_model --agent1-model data/models/best_a2c_v11.npz
uv run python scripts/build_model_report.py
```

Sonda riproducibile di simmetria dei semi:

```bash
uv run python scripts/probe_suit_symmetry.py \
  --model data/models/best_a2c_v13.npz \
  --out-json docs/reports/evidence/suit_symmetry_v13.v1.json
```

Policy simmetrizzata, costo e direct match:

```bash
uv run python scripts/benchmark_suit_symmetrized.py \
  --model data/models/best_a2c_v13.npz --observations 160 --decisions 10000
uv run python scripts/evaluate_agents.py --benchmark medium --engine domain \
  --agent0 bc_model_suit_symmetrized --agent0-model data/models/best_a2c_v13.npz \
  --agent1 bc_model --agent1-model data/models/best_a2c_v13.npz
```

Avvio locale:

```bash
briscola-server --reload
```
