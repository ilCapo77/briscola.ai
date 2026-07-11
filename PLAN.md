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
- Runtime web zero-Numba: dominio, search PIMC e solver usano Python nel processo web; Numba resta per training,
  valutazioni e benchmark. In produzione: FastAPI Cloud, Redis per stato/pub-sub, Postgres in modalità `dataset`.
- Il catalogo modelli espone v13 come policy compatibile e non espone value/belief (`value_mlp_v1`/`belief_mlp_v1`),
  che sono asset interni. Policy v10/v11/v13, value e belief necessari al runtime sono tracciati in Git; gli altri
  `.npz`, dataset e benchmark restano locali e gitignored.
- L'event log live contiene `human_action`, `ai_action`, `game_finished`, consenso e metadati modello.
  `export_live_actions.py` produce la sequenza unica umano+IA e può filtrare versione, agente, modello e bot.
- Il diario pubblico e gli approfondimenti tecnici arrivano al capitolo 19,
  `docs/diario/19-la-copia-non-basta.md`.

## Prossima Decisione

La prima diagnostica riproducibile ha identificato una leva concreta: v13 attribuisce significato ai nomi arbitrari
dei semi. La duplicazione paired del loss A2C non l'ha corretta e ha introdotto un piccolo costo di forza. Non basta
quindi mostrare alla rete la stessa traiettoria rinominata: serve un obiettivo che chieda direttamente output
coerenti, senza trattare come on-policy un'azione campionata da un'altra distribuzione. Ipotesi e criteri go/stop
delle sette piste restano in `docs/plans/prossima-iterazione-modello.md`.

1. **Ablation di consistency loss sui semi.** Usare la trasformazione numerica già verificata, ma lasciare invariato
   il loss A2C on-policy: aggiungere separatamente una cross-entropy/KL tra la policy sull'osservazione originale
   (target con stop-gradient) e la policy sulla copia rinominata, dopo aver riallineato le 40 azioni. Prima fare un
   test di gradiente che dimostri riduzione della divergenza su un batch congelato; poi screening breve a tre seed e
   piccola griglia del coefficiente. Gate: calo netto di flip e JS, nessuna regressione policy-only; solo allora PIMC
   16×8 e decision quality. Il paired A2C v0 è chiuso e non va semplicemente allungato.
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

Avvio locale:

```bash
briscola-server --reload
```
