# Piano operativo — Briscola AI

> Questo file è VOLUTAMENTE breve: fotografa lo stato reale e le prossime azioni.
> I dettagli storici vivono nei messaggi di commit, in `docs/plans/` (ipotesi, gate e
> risultati) e nel diario di bordo (<https://ai.briscola.dev/diario> + `docs/diario/`).

## Stato Corrente

- Versione: `0.31.1` (2026-07-07). In 0.30.0: sonda trump_saver (dominio+fast/numba),
  cold start FastAPI Cloud 19s→13.7s (warm-up/provisioning in background, modelli
  committati nell'immagine). In 0.31.0: promozione v11, default UI =
  `bc_model_pimc_belief_16x8` con search PYTHON. In 0.31.1: **runtime web ZERO-NUMBA** —
  anche il solver endgame è python (`solve_endgame_fast`, 0.09 ms/chiamata) con playout
  della principal variation nei rollout (una soluzione per rollout invece di una per
  carta: stesso delta minimax, esiti IDENTICI al bit su 400 partite a parità di seed).
  Costo: 16.6 vs 17.0 ms/mossa del numba (pari). Guadagni: niente warm-up JIT, −47 MB
  RSS per replica (116→69), import app 0.23s. I kernel numba restano per training e
  benchmark (import PIGRO via PEP 562 in `ai/endgame/__init__`).
- Produzione: <https://ai.briscola.dev> (FastAPI Cloud, stato su Redis, realtime pub/sub,
  event log Postgres in modalità `dataset` con eventi `ai_action` auditabili).
  **Scale-to-zero dopo ~90s di idle** (misurato dai log): il cold start è frequente.
- Modello consigliato: `best_a2c_v11.npz` (encoder **v4**, hidden 256, SENZA overkill
  guard: misurato dannoso su v11, −0.5). Promosso v0.31.0 — ipotesi dose-shift: 40% del
  cartellone al maestro PIMC 16×8 belief (dose tolta al VL), base/maestri v10, **5M
  partite (6× meno di v10)**: **+0.85 su v10** (CI coppie +0.71..+0.99) e **+20.80 su
  heuristic_v1** (record assoluto, era 20.52). La curva dei rendimenti (+2.46 → +0.97 →
  +0.66) si è RIALZATA: conta la dose/qualità del maestro, non il volume.
- Avversari avanzati selezionabili: **`bc_model_pimc_belief_16x8`** (default: +3.36 su
  v11+solver, CI +3.05..+3.66, ~15 ms/mossa pensata, search python),
  `bc_model_pimc_belief_64x10` (il più forte: +3.9, ~73 ms/mossa in python),
  `bc_model_value_lookahead_8x8`, `bc_model_pimc_16x8`.
- **PRIMA del prossimo deploy (CRITICO)**: rimuovere dal cloud le env
  `BRISCOLA_MODEL_URL`/`BRISCOLA_MODEL_SHA256` (+ le coppie VALUE/BELIEF): pinnano la v10
  e, col nuovo `DEFAULT_MODEL_ID=best_a2c_v11.npz`, il mismatch SHA farebbe riscaricare
  la v10 SOPRA il file v11 committato. I tre asset sono nell'immagine: il provisioning
  serve solo come fallback.
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

### 0. Postura corrente (2026-07-08): ASCOLTO, non training

v12 ha dimostrato che la policy reattiva è **satura** (+0.11 su v11 con 10M partite ed edge
maestro intatto): il prossimo punto NON si compra con più partite/più maestro. Decisione:
nessun run finché non c'è una nuova ipotesi con un bersaglio misurato. La sorgente di
ipotesi è il **campo** — v11 è in produzione col default nuovo e l'event log registra.
Quando ci saranno ~50-100 partite umane complete contro v11, ripetere l'audit (script
pronti) e misurare il contatore carichi-guidati/persi SUGLI UMANI: decide se il buco vale
un ciclo o è marginale.

**Pagella della nonna (2026-07-08, `scripts/behavior_profile.py`, profilo v11 in
`data/behavior_profile_v11_20260708.json`)**: profilo comportamentale di v11 su 2k partite
strumentate × 3 avversari (saver/specchio/h1). Risultati: apertura liscia ~39% (mediocre),
carichi guidati ~10-13% (71% persi vs conservatori — il vizio noto), briscola su piatti
poveri ~6% (spreco noto), punti regalati per presa persa **0.97 (ottimo)**, scarto dal seme
corto ~75% (**"sbianchirsi" appreso da solo, sorpresa positiva**), asso di briscola tenuto
fino alla presa ~17 (rispettata). **Scoperta forte: "cavare le briscole" con mano lunga
(≥4) = 0.0% ASSOLUTO su tutti e tre gli avversari** (vs 18% con mano corta): condizionamento
INVERTITO rispetto alla regola della nonna, secondo vizio sistematico dopo i carichi. Il
profilo è quasi identico sui 3 avversari → conferma che la policy è INCONDIZIONALE (non
cambia stile con l'avversario), che è la radice del problema.

**Ipotesi evolutiva principale (in progettazione): encoder v5 con "feature di stile".**
4-6 contatori da informazione pubblica (frequenza taglio carichi dell'avversario, % aperture
lisce, briscole spese su piatti poveri, punti rifiutati…) appesi alle feature: danno alla
policy la materia prima per comportarsi diversamente con avversari diversi — oggi
matematicamente impossibile. Stesso schema validato con le feature v4 (+0.27). Gate con
firma inequivocabile: il contatore carichi deve scendere **contro il saver ma non contro lo
specchio** (adattamento, non media). Scavalca in priorità il potential-shaping (v13), che
cura solo il vizio piccolo (spreco); tenere entrambi.

**Due sonde da arbitrare** (giudice = PIMC, come le altre): (a) la cavata delle briscole con
mano lunga — la nonna ha ragione o v11 ha scoperto che in 1v1 non paga? (b) l'asso di
briscola giocato prima (giudice-search) vs tenuto fino in fondo (umani vincenti).

### 1. Post-Deploy 0.25.0

Monitorare la CPU delle repliche nei primi giorni: il default PIMC-belief costa
~25–50× per mossa rispetto a v8 puro anche con la search JIT (~37 ms/mossa pensata).
Se il traffico lo rende oneroso, tornare a default `bc_model` è un one-liner nel frontend.
Opzionale: registrare il sito su Google Search Console e inviare la sitemap.

### 2. Monitoraggio Produzione E Audit

- Classificare le mosse sospette dal `decision_type` dell'event log: `fallback` → policy
  base; `solver` → ricostruzione/endgame; `lookahead`/`search` → value/PIMC/guard.
- Errori ricorrenti → nuova ipotesi misurabile, non fix estemporanei.
- Non usare dati umani per training finché volume, consenso, qualità e privacy non sono
  riverificati.

**Primo audit dei dati di campo (2026-07-07)** — dettaglio completo (metodo, verdetti,
numeri) in `docs/plans/audit-campo-2026-07-07.md`; in sintesi: 123 partite umane complete,
40 vs v10 con 7 vittorie umane (17.5%, in linea con l'atteso). Le 7 vittorie: zero errori
interni dell'IA (giudice 128×5 + solver), ma **bias di famiglia** comportamentale — 8/9
carichi guidati persi (~111 pt) contro umani che conservano le briscoline per tagliare
(finestra fallback deck 22→8). Lo stile è codificato in **`heuristic_trump_saver`**
(sonda di exploitability, solo dominio, registry ma non UI): rule-based più forte del repo
(+10.47 su h1) ma **exploit differenziale NON confermato** (vs v10 −13.79, peggio della
transitività ~−10.4: il campione delle 7 vittorie era selezionato). La sonda resta la
baseline anti-regressione del bias: −13.79 è il numero da battere in differenziale.
Artefatti: `benchmarks/experiments/trump_saver/`, `data/field_audit_20260707/`.

### 3. Hardening Continuo

Aggiungere test solo su casi reali sospetti o quando si toccano regole/observation/search.

**Robustezza ai transitori (2026-07-06, v0.28.1→v0.29.1)**: dopo l'incidente di produzione
del 2026-07-06 (Redis+Neon irraggiungibili ~12 min), l'app ha retry/degrado a ogni strato:
riconnessione WS client con backoff+health check (fix zombie handler), riconciliazione
delle mosse su blip di rete (resync dalla versione server, mai errore crudo al giocatore),
retry Redis dentro i comandi (3 tentativi, ~0.5s) + health check pool, endpoint WS che
degrada con 1013 se lo store e' giu'. Notifiche email errori via Mailgun pronte ma NON
configurate in prod (opt-in via env). Primi test e2e JS (Playwright).

**Cold start FastAPI Cloud (2026-07-07, CHIUSO con misure)**: scale-to-zero dopo ~90s di
idle; il risveglio costava ~18.7s (10.2s di warm-up Numba sincrono + download modelli).
Fix in 0.30.0/0.31.0: provisioning+warm-up in background con telemetria per fase, tre
`.npz` di runtime committati nell'immagine, search PIMC tornata python (resta solo il
solver JIT, ~2s in bg). Esito misurato: **TTFB del risveglio 13.7s, IDENTICO prima e dopo
il de-JIT** → il pavimento è della PIATTAFORMA (scheduling container + boot + cadenza del
probe di readiness), non dell'app. Il guadagno vero: replica subito operativa al 100%
(niente 10s di compilazione mentre serve i primi giocatori) e default ~2.4× più capiente.
Per abbattere i 13.7s restano solo: keep-alive esterno con ping <90s (nota: cron GitHub
Actions ha minimo 5 min → serve un job che pinga ogni 60s al suo interno), piano a
pagamento, o messaggio "sto svegliando il server" in UI (cosmetico ma onesto).

**Load test infrastruttura (2026-07-06, `scripts/loadtest/`)**: bot Locust che giocano
partite complete via REST. Risultato su prod (free tier): con `bc_model` 10 giocatori
simultanei = 0 errori ma latenza mossa 2x (692ms medi); con l'avversario di default
(PIMC belief 64x10) 10 partite simultanee saturano le repliche (3.5% di 502/503 per
~90s, poi autoscale recupera). **Capacita' reale col default: ~5-8 giocatori simultanei.**
Leva futura se serve: default a 32x8 (~2x capacita', costo in forza da misurare).
Le partite bot si escludono dal dataset con `WHERE client_id != 'loadtest-bot'`.

Debito minore residuo (nessuno urgente; prenderlo quando si tocca l'area):

- DX script: `scripts/_common.py` condiviso e `logging` al posto di `print`;
- AI: RNG serial vs parallel non riproducibile cross-`workers` in `decision_quality`;
- frontend: CSS monolitico (~900 righe).

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

**Sonda edge maestro su v10 (2026-07-07)**: PIMC belief su BASE v10 vs v10+solver, 4.000
seat-fair (stessa ricetta della conferma v0.23.0): 16×8 **+3.37** (CI +3.06..+3.69,
6.2 ms/mossa pensata), 64×10 **+3.87** (CI +3.51..+4.23, 36.7 ms). Edge INVARIATO rispetto
a base v8 (+3.66): lo sparring non assorbe il vantaggio strutturale della search (media sul
rumore del mazzo). Il ramo PIMC-as-teacher è quindi USCITO dal frigo: **run v11 lanciato
2026-07-07** (A2C 5M, base/maestri v10, PIMC 16×8 belief al 40% del cartellone spostando
dose dal VL, iperparametri v10, seed 20260707, `--metrics-mode summary`); throughput
osservato ~26k partite/min (checkpoint 1M a +38'). Al termine: gate big seat-fair vs v10
(successo = +0.3..+0.5; sotto +0.2 il ramo sparring si chiude).

**RISULTATI v11 (2026-07-07, run completato in 3h33m, gate big 100k seat-fair numba,
artefatti `benchmarks/experiments/fase3/v11_vs_*.json`):**
- **vs v10: +0.85 (CI +0.71..+0.99)** — SOPRA la banda di successo (+0.3..+0.5) e sopra
  il +0.66 di v10-su-v9: la curva dei rendimenti decrescenti (+2.46 → +0.97 → +0.66) si è
  RIALZATA. L'ipotesi dose-shift è validata: spostare dose dal maestro consumato (VL) a
  quello intatto (PIMC 16×8) rende più di 6× il volume (v11: 5M partite; v10: 30M).
- **vs heuristic_v1: +20.80 (CI +20.62..+20.97)** — nuovo record assoluto (era +20.52).
- **vs trump_saver: +14.34** (v10: +13.79, misurato però su medium 10k domain) —
  miglioramento proporzionale alla forza generale, nessun guadagno differenziale sul
  fianco "umano": atteso, la sonda non era nel cartellone. Resta l'ipotesi v12.
- Nota per la promozione: nel gate v11 è girato SENZA `overkill_guard` (il metadata
  `inference_overkill_guard` non risulta nel `.npz`), v10 CON: prima di promuovere,
  normalizzare il guard e riconfermare il gate 1. Ramo sparring: APERTO (prossimo giro
  possibile: maestro PIMC su base v11, e/o trump_saver nel cartellone).

**Ipotesi v12 (dopo il gate v11)**: diversità di stile nel cartellone — `heuristic_trump_saver`
come avversario di training (dose 10–15% dalla quota bar) per curare il bias di famiglia sui
carichi guidati. Tre termometri: (1) differenziale vs trump_saver (baseline v10 = −13.79);
(2) big vs v10/v11 (no regressioni); (3) contatore briscole spese su piatti ≤2 punti.
Prerequisito FATTO (commit d2888be): sonda tradotta in fast+numba con parità ESATTA a tre
motori (il +10.47 vs h1 riprodotto identico campo-a-campo in `--engine fast`), utilizzabile
in `--opponent-mix` (smoke train verificato); corretto anche un range check hardcoded nel
collector A2C numba che rifiutava codici agente nuovi.
**Run v12 COMPLETATO (2026-07-08, 10M partite, seed 20260708)**: mix
`bc_model:0.15, pimc_belief 16x8:0.40, VL_8x8:0.20, trump_saver:0.12, h1:0.04, h2:0.06,
random:0.03`, base/maestri v11. **ESITO NEGATIVO, niente promozione** (artefatti
`benchmarks/experiments/fase3/v12_vs_*.json`): vs v11 +0.11 (CI −0.03..+0.25, n.s.);
vs trump_saver +14.80 (vs 14.34 di v11: +0.46 differenziale, in parte familiarità col
partner deterministico); vs h1 +20.74 (record invariato). Contatore comportamentale
(2k partite strumentate vs saver): carichi guidati 11.1% vs 10.4%, persi 71.3% vs 71.4%,
briscole su piatti poveri 6.7% vs 6.8% — **comportamento IDENTICO a v11**. Due diagnosi
da tenere: (a) il maestro non si consuma ma l'ALLIEVO SI SATURA (+0.85 → +0.11 con edge
maestro identico a +3.36); (b) la punizione diluita non sposta un comportamento che paga
in media — guidare carichi costa ~−11 attesi contro il saver (12%) ma paga contro il
resto del cartellone (88%): una policy incondizionale fa la media e resta ferma. Il vizio
n.1 (carichi vs conservatori) è intrinsecamente CONDIZIONALE → richiede riconoscimento
dello stile avversario dalla storia (Fase 4) o resta coperto dalla search a runtime.
v12 resta come artefatto locale non promosso.

**Ipotesi v13 (in coda, scelta dal maintainer 2026-07-07): reward shaping POTENTIAL-BASED
sull'economia di briscola.** Diagnosi: lo spreco di briscoline è la lezione che l'A2C
impara peggio — costo ~2-3 punti che arriva molte prese dopo, annegato in ±28 di rumore
del mazzo (problema di segnale/rumore, non di orizzonte: 20 decisioni). Cura: potenziale
Φ(stato) = valore pesato delle briscole ancora in mano; reward per-mossa += ΔΦ. La somma
telescopica non cambia il ritorno totale (Ng 1999, policy-invariant con γ=1 e Φ(terminale)
=0): non dice COSA fare, avvicina solo il segnale alla mossa. È la versione matematicamente
corretta delle DUE penalità anti-spreco fallite nella storia del progetto (che alteravano
l'obiettivo). Trigger: se il contatore comportamentale di v12 (briscole spese su piatti
≤2 punti; carichi guidati e persi vs conservatori) mostra il buco ancora aperto. Gate:
i soliti quattro di v12. Caveat: Φ incorpora un giudizio nostro sui pesi → rimisurare a
ogni generazione come l'overkill guard (utile a v6, dannoso a v11).
Razionale completo (difesa ≠ imitazione, dosaggio condizionale, leve alternative scartate)
in `docs/plans/audit-campo-2026-07-07.md` §7.

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
