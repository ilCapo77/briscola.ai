# Piano: Belief → Determinizzazioni Pesate → Expert Iteration

> Piano di ricerca dettagliato per la progressione del modello dopo `best_a2c_v7`.
> Fonte: indagine approfondita del 2026-07-02 (analisi encoder/training/esperimenti falliti + letteratura).
> `PLAN.md` resta la fonte di verità operativa: questo documento è il dettaglio dell'ipotesi.

## 1. Perché siamo a un punto morto (diagnosi)

Il ristagno post-v7 è **sovradeterminato**: tre soffitti indipendenti si rinforzano a vicenda. Ogni
esperimento recente (distillazione PIMC, value retrain v1 e leaf-pairwise, v7 stesso) ha sbattuto
contro almeno uno di essi.

### 1.1 Soffitto informativo (il più grave)

L'osservazione è **aggregata-markoviana**: `seen_cards_onehot` / `out_of_play_cards_onehot` sono
bitmask globali sulle 40 carte, senza ordine né attribuzione. `make_player_observation`
(`domain/observation.py`) comprime `p.captured_cards` — che nel dominio È già segmentato per
giocatore — in due bitmask, cancellando:

- la storia ordinata delle prese (chi ha giocato cosa, in che ordine);
- chi ha vinto ogni presa e quanti punti valeva;
- la sequenza comportamentale dell'avversario (es. "non ha mai tagliato a coppe"),
  che è l'informazione centrale per inferirne la mano.

Conseguenza: **l'opponent modeling comportamentale è irrappresentabile con questi input, per
qualsiasi rete di qualsiasi capacità**. Non è un vincolo anti-cheat (tutto ciò che manca è stato
pubblicamente osservato): è una perdita introdotta dall'encoding.

### 1.2 Soffitto di rappresentazione

La policy è una MLP `310 → 128 → 40` (~45k parametri, 1 hidden layer, ReLU), **identica per 5
generazioni** (v3..v7). Input flat one-hot: nessun embedding di carta, nessuna condivisione tra
semi, nessun meccanismo relazionale. Il blocco v3 dell'encoder (22 feature hand-crafted) esiste
proprio per iniettare a mano le relazioni che l'architettura non può derivare. Il value model
(`310 → 128 → 1`) condivide encoder e architettura.

### 1.3 Soffitto del segnale di apprendimento

- **Catena warm-start senza controllo**: `bc_v3 → v3 → v4 → v5 → v6 → v7`, ogni generazione
  inizializzata dalla precedente, iperparametri congelati (lr 3e-4 costante, entropy 5e-4,
  γ=1.0, update_every 20, hidden 128). Anchor CE verso `bc_v3` (β=0.01) per v3–v6: la ricerca è
  confinata nel bacino del teacher BC originale. **Non esiste alcun run from-scratch di
  confronto** a scala reale.
- **A2C Monte-Carlo puro**: advantage = return non scontato − V(s), nessun GAE/n-step, nessuna
  normalizzazione dell'advantage, nessun gradient clipping, value head lineare su trunk condiviso.
  Alta varianza: gran parte delle 5M partite/generazione paga la varianza, non l'apprendimento.
- **Avversari congelati della stessa famiglia**: v7 è allenato contro un unico opponent statico
  (`bc_model_value_lookahead_8x8` su base v6). Risultato documentato in PLAN.md: v7 batte il
  target statico ma NON supera la search (`v7+solver` vs `v6+solver` = −0.64). La ricetta ha
  saturato ciò che può estrarre **senza search nel loop di training**.
- **Nessun gating intra-run**: si promuove l'ultimo checkpoint del run, non il migliore.

### 1.4 La rilettura del fallimento della distillazione (il dato che inchioda)

Il "val acc 57%" della distillazione PIMC è misurato **sul subset di disaccordo** (PIMC ≠ base),
dove la chance è ≈50% (restano ~2 alternative plausibili): **è un coin-flip**. Evidenza raccolta:

- **capacità esclusa**: hidden=512 memorizza il train senza migliorare la val;
- **nitidezza label esclusa**: i soft-label peggiorano scaldando (T=2→56.9%, T=5→55.6%, T=10→52.8%);
- **causa primaria — non-identificabilità**: la correzione PIMC è un integrale sulle mani
  avversarie nascoste; non è una funzione dell'osservazione v3;
- **cause aggravanti**: covariate shift BC classico (dataset on-policy del base, mai del
  candidato: no DAgger) e teacher disallineato (probe teacher-v6 su base-v7 perde);
- **rumore reale ma secondario**: solo 2935/18184 record search sono strong/reliable
  (`margin≥2 ∧ ci_low≥0`); l'argmax su 16–64 determinizzazioni è instabile proprio dove serve.

Stesso schema per i value retrain (v1, leaf-pairwise): metriche offline su (medie dominate dai
root facili), A/B runtime nullo/negativo (deciso dalle poche decisioni pivotali a basso margine).

### 1.5 Il margine esiste

`PIMC(v6,16×8)+solver` batte `v6+solver` di **+3.59** (CI95 +2.48..+4.70). La search sa giocare
meglio della policy reattiva di ~3-4 punti/partita: il problema non è il tetto del gioco, è che
nessuna delle strade tentate riesce a travasare quel vantaggio nella policy.

### 1.6 Un dettaglio tecnico che sblocca la Fase 3

`pimc.py` calcola la matrice `per_determinization_scores[det][azione]` (righe ~449, ~497) ma
**non la persiste**: la diagnostica salva solo media per azione + margine best-vs-second. Per
target soft/distribuzionali affidabili serve persisterla.

## 2. Obiettivo e definizione di "progresso deciso"

- **Obiettivo primario**: una policy reattiva `.npz` (o un agente runtime a pari CPU) che superi
  `bc_model_value_lookahead_8x8` (l'attuale campione runtime) nel confronto seat-fair con CI
  su coppie — cioè assorbire nella policy il vantaggio oggi esclusivo della search.
- **Obiettivo secondario**: alzare il tetto dell'agente search stesso (belief-weighted PIMC).
- **Scala attesa**: la sonda dell'oracolo (Fase 0) fissa il target numerico. Ordine di grandezza
  ragionevole: +3..+6 punti/partita vs v7, contro i +2.3 dell'ultimo giro incrementale.
- Ogni fase ha criteri di kill espliciti: se una fase fallisce il suo gate, si ferma quel ramo,
  non si "insiste con più dati simili" (lezione dei value retrain).

## 3. Fase 0 — Sonde diagnostiche (1-2 giorni, quasi solo script esistenti)

Prima di investire, tre misure che cambiano le priorità:

### 0.a Sonda di exploitability
Allena un best-response A2C (ricetta identica, 1-2M partite) **contro v7 congelato**
(`train_a2c.py --opponent bc_model --opponent-model best_a2c_v7.npz`).
- Lettura: se `BR vs v7 >> v7 vs v6` (es. BR vince di +6 o più), il self-play converge a
  strategie sfruttabili → la diversità degli avversari (league/population o ExIt) sale di priorità.
- Costo: una notte di CPU. Nessun codice nuovo.

### 0.b Sonda dell'oracolo (tetto realistico)
`PIMC(v7, 64×10)+solver` vs `v7+solver`, 2-4k partite seat-fair (CI su coppie).
- Lettura: il margine misura quanto vale, al massimo, "portare la search nella policy" con
  l'attuale qualità di determinizzazione. Se il margine con budget 4x resta ~+3-4, quello è il
  target della Fase 3; se cresce molto, il soffitto è più alto del previsto.
- Costo: qualche ora di CPU. Nessun codice nuovo.

### 0.c Controllo from-scratch
`train_a2c.py` con ricetta v7 ma **senza `--init`** (e senza anchor), 5M partite.
- Lettura: se from-scratch ≈ v7, la catena warm-start non è il problema (e si smette di
  sospettarlo); se from-scratch < v7 di molto, il warm-start è un asset da conservare; se
  from-scratch > v7, la catena era un ottimo locale (improbabile ma va escluso).
- Costo: una notte di CPU. Nessun codice nuovo.

Gate di fase: nessuno blocca le fasi successive; i risultati **orientano** (0.a → quanta
diversità di avversari serve; 0.b → target numerico; 0.c → init della Fase 4).

### Risultati Fase 0 (eseguita 2026-07-02)

Nota di costo: grazie al throughput Numba (~4.150 partite/s per il training, 0,28 s/partita per
PIMC 64×10) l'intera fase è costata **~45 minuti di CPU**, non "notti". Le stime di effort delle
fasi successive vanno lette di conseguenza: il collo di bottiglia di ExIt sarà la search del
teacher (~0,08 s/decisione search), non il training.

**0.a Exploitability — v7 è quasi inespugnabile nella sua classe.**
Best-response A2C (2M partite, init=v7, opponent=v7 congelato, seed 20260702) vs v7,
10k seat-fair: **+0.70** (CI95 coppie +0.22..+1.19), score rate 0.511.
Un avversario allenato SPECIFICAMENTE contro v7 lo batte di meno di un terzo del margine
v7-su-v6 (+2.46). Lettura: il self-play non sta convergendo a strategie sfruttabili;
league/population play NON è la leva; la classe "MLP 128×1 su encoder v3" è satura attorno a v7.

**0.b Tetto oracolo — la search vale ~+3.8, e satura con il budget.**
`PIMC(v7, 64×10)+solver` vs `v7+solver`, 4000 partite seat-fair: **+3.76**
(CI95 coppie +3.40..+4.12), score rate 0.564. Con budget 4× rispetto al runtime (16×8) il
margine cresce solo marginalmente rispetto al +3.59 storico → la search a determinizzazioni
UNIFORMI satura intorno a +3.8: per alzare il tetto serve migliorare la qualità delle
determinizzazioni (belief, Fase 2), non il loro numero. Dettaglio: le mosse search sono il
17,5% delle decisioni (14k/80k) — tutto il vantaggio vive nella finestra di fine partita.
Artefatto: `benchmarks/experiments/fase0/oracle_pimc_v7_64x10_vs_control_4k.json` (locale).

**0.c From-scratch — la catena warm-start è un asset, non una trappola.**
Ricetta v7 identica (opponent value-lookahead, 5M partite, seed 20260702) ma init casuale:
vs v7 = **−5.42** (CI95 coppie −5.91..−4.92); vs `heuristic_v1` = +12.58 (v7 fa +18.73).
A parità di budget il from-scratch arriva a livello ~v2/v3. Lettura: il valore accumulato nella
catena è reale; non c'è evidenza di ottimo locale indotto dal warm-start. Implicazioni:
ExIt (Fase 3) parte da `policy_0 = v7`; l'eventuale from-scratch profondo della Fase 4 richiederà
più budget e/o curriculum, non è gratis.

**Sintesi Fase 0**: le tre sonde convergono sulla stessa conclusione del piano — l'unico
margine dimostrato (+3.8) vive nella search ed è irraggiungibile dall'interno della classe
reattiva attuale (0.a); il budget di search non lo alza (0.b); e non c'è scorciatoia
"ripartire da zero" (0.c). Restano esattamente le leve delle Fasi 1-3: informazione (encoder
v4), qualità delle determinizzazioni (belief) e trasferimento iterato (ExIt).

## 4. Fase 1 — Encoder v4: restituire la storia (prerequisito di tutto)

### 4.1 Modifiche a `PlayerObservation` (dominio)

Aggiungere campi **pubblicamente osservabili** (anti-cheat per costruzione):

- `trick_history`: tupla ordinata di prese completate, ognuna
  `(cards_in_order: tuple[(card_id, player_index)], winner_index, points)`;
- eventualmente `draw_order_known`: nulla di nuovo da esporre (l'ordine di pesca non è pubblico),
  esplicitamente NON incluso.

Nota di fattibilità: `GameState.players[i].captured_cards` conserva già le carte per giocatore;
serve arricchire `engine.step` perché registri la presa completa nell'ordine di gioco (oggi
l'ordine intra-presa è ricostruibile solo dal tavolo corrente). Aggiornare serializzazione
(bump `SERIALIZATION_SCHEMA`), fast path e test di parità.

### 4.2 Feature v4 (`observation_encoder.py`)

Blocco aggiuntivo rispetto a v3 (dimensioni indicative):

- **Conteggi comportamentali per-avversario** (il cuore): per ogni seme, quante carte l'avversario
  ha giocato di quel seme; quante volte ha tagliato di briscola; quante volte ha risposto al seme
  di uscita vs scartato; punti che ha ceduto su lead bassi (~16-24 feature);
- **Ultime K prese** (K=3-5): per presa, one-hot compatti di (carta lead, carta risposta, chi ha
  vinto, punti/11) (~K×(8-12) feature);
- **Traiettoria punti**: punti per presa recenti invece del solo cumulativo (~K feature);
- fix igiene v1: normalizzare i blocchi `hand_points`/`hand_strength` (oggi grezzi 0-11 contro
  scalari /120 — asimmetria di scala documentata).

Vincoli: parità dict/oggetto (come v2/v3), traduzione nel fast path + kernel Numba
(`numba/observation.py`), test di parità nuovi. Il contratto `feature_dim_for_encoder_version`
propaga v4 automaticamente a policy e value model.

### 4.3 Gate di fase

- Retrain BC/A2C **a parità di tutto il resto** (128×1, ricetta v7) su encoder v4:
  atteso un guadagno anche piccolo ma positivo vs v7 (holdout + CI su coppie).
- Kill: se v4 a parità di ricetta è ≤ v3 E la belief (Fase 2) su v4 non supera nettamente la
  belief su v3, il design delle feature va rivisto prima di procedere.
- Stima effort: 3-5 sessioni di lavoro (dominio+engine, encoder, fast/numba, test parità).

## 5. Fase 2 — Belief network e determinizzazioni pesate

### 5.1 La rete belief

`B(observation_v4) → P(carta c in mano avversaria)` per le 40 carte (sigmoid multi-label,
mascherata sulle carte non-ignote: per le carte in `seen`/mano propria la label è nota).

- **Dataset**: self-play numerico (riuso di `generate_value_dataset_numba`-style): input =
  osservazione, label = mano avversaria vera dal full-state. Lecito: il full-state è usato SOLO
  per le label a training; a inference la rete vede solo l'osservazione.
- **Architettura iniziale**: stessa MLP (v4_dim → 128/256 → 40) per coerenza con l'infrastruttura.
- **Gate offline**: log-loss e top-k recall vs baseline uniforme-sulle-ignote. Se la belief non
  batte nettamente l'uniforme, l'encoder v4 non porta segnale comportamentale → tornare a 4.3.

### 5.2 Uso 1 — Determinizzazioni pesate (attacca subito il campione runtime)

Oggi `determinize_observation` campiona le mani avversarie **uniformemente** tra le carte ignote.
Sostituire con campionamento ∝ belief (senza rimpiazzo, rinormalizzando), in PIMC e
value-lookahead. Precedente diretto in letteratura: *Policy Based Inference in Trick-Taking Card
Games* (Rebstock, Solinas, Buro, Sturtevant 2019, arXiv:1905.10911) migliora Kermit (SOTA Skat)
esattamente con questo meccanismo.

- **Gate runtime**: `PIMC-belief(16×8)+solver` vs `PIMC-uniforme(16×8)+solver`, 4k+ seat-fair,
  CI su coppie. Idem per value-lookahead.
- Kill: se il pesatura non migliora (CI compatibile con zero) con belief offline sana, indagare
  la calibrazione della belief prima di buttare l'idea (il campionamento è sensibile alle code).
- Stima effort: 2-3 sessioni (dataset+training belief, sampling pesato, eval).

### 5.3 Uso 2 — Input/auxiliary per la policy

- Concatenare l'output della belief (40 valori) alle feature della policy, oppure
- testa ausiliaria belief nel training A2C (loss ausiliaria che plasma il trunk condiviso).
- Gate: A2C a parità di ricetta con/senza belief input, holdout + CI coppie.

## 6. Fase 3 — Expert Iteration (il volano)

Sostituire la distillazione one-shot con il loop iterato (Anthony/Tian/Barber, NIPS 2017 — il
principio di AlphaZero):

```
policy_0 = v7 (o from-scratch v4, decisione informata dalla Fase 0.c)
repeat k:
  expert_k   = PIMC(policy_k, belief_k) + solver         # più forte della policy per costruzione
  dataset_k  = partite AVANZATE DA expert_k (o dal candidato: DAgger)   # niente covariate shift
               target = distribuzione soft dagli action_values per determinizzazione
  policy_k+1 = train su dataset_k (BC/A2C misto) partendo da policy_k
  belief_k+1 = retrain belief su self-play di policy_k+1
  gate_k     = policy_k+1 vs policy_k E expert_k+1 vs expert_k (seat-fair, CI coppie)
```

Correzioni specifiche ai fallimenti documentati:

1. **Covariate shift** → gli stati vengono dalle partite dell'expert/candidato, non del base
   (`advance_with_teacher=True`, già supportato da `generate_pimc_teacher_dataset.py`).
2. **Teacher disallineato** → il teacher è SEMPRE la policy corrente (mai v6 su base v7).
3. **Rumore label** → persistere `per_determinization_scores` in `PIMCSearchDiagnostics`
   (modifica a `pimc.py`); budget adattivo: più determinizzazioni solo dove il margine è
   incerto; target soft dalla distribuzione, pesati per affidabilità.
4. **Non-identificabilità** → mitigata da encoder v4 + belief input (Fasi 1-2): il target
   diventa molto più funzione dell'input osservabile.

- **Gate per iterazione**: entrambe le curve (policy e expert) devono salire; 2 iterazioni
  consecutive piatte = stop e analisi.
- **Criterio di successo del piano**: `policy_K` (reattiva) ≥ `bc_model_value_lookahead_8x8`
  attuale, oppure `expert_K` > campione attuale di un margine ≥ sonda oracolo/2.
- Stima effort: 3-4 sessioni per l'harness del loop (molti pezzi esistono già:
  teacher dataset, soft-label in `train_bc`, warm-start) + run multi-notte.

## 7. Fase 4 — Architettura (dentro ExIt, non da sola)

Quando il loop ExIt è in piedi (o al più tardi alla prima iterazione piatta per capacità):

- MLP più profonda (2-3 hidden, 256-512) con **embedding di carta** condivisi (40×d) e
  aggregazione delle feature per-carta, opzionalmente encoder della storia (stile DouZero:
  LSTM/attenzione sulle giocate — DouZero, ICML 2021, arXiv:2106.06135, ha dominato DouDizhu
  con Deep MC + storia encodata, from scratch).
- **Costo reale da pianificare**: i kernel Numba (`numba/observation.py`, `numba/mlp.py`)
  hardcodano il forward a 1 hidden layer; servono kernel generalizzati (o un forward a profondità
  fissa 2-3). Da fare una volta, riusato ovunque.
- Gate: from-scratch v4-deep dentro ExIt vs v4-128×1 dentro ExIt, a parità di budget partite.

## 8. Cosa NON fare (kill già decisi)

- **Niente altri fine-tune del value model** su dataset simili: tre fallimenti (v1,
  leaf-pairwise v6 e v7) con causa strutturale ora identificata (gate offline dominato dai casi
  facili; runtime deciso dalle code).
- **Niente v8 warm-start contro avversario congelato**: v7 dimostra che si impara a battere il
  target statico senza superare la search.
- **Niente aumento di capacità sulle feature v3**: escluso sperimentalmente (hidden=512).
- **Niente lavoro sul 4-player o su PPO/GAE** finché questo asse non è concluso (PPO resta
  l'ottimizzazione della Fase 3/4 SE la varianza si rivela il collo di bottiglia, non prima).

## 9. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Encoder v4 rompe la parità fast/numba | test di parità nuovi PRIMA del training (pattern test-àncora già in uso) |
| Belief mal calibrata peggiora il sampling | gate offline con calibrazione (reliability diagram), fallback misto uniforme+belief (λ) |
| ExIt amplifica bias della search (strategy fusion/non-locality di PIMC — Long et al., AAAI 2010) | la belief riduce l'errore di determinizzazione; monitorare con decision-quality; valutare idee EPIMC (arXiv:2408.02380) solo se emerge il problema |
| Costo CPU del loop | budget per iterazione fisso (es. 200-500k stati teacher); il collo è la search, non il training |
| Regressione stile di gioco (overkill ecc.) | gate decision-quality vs `heuristic_v1` già standard per ogni promozione |

## 10. Riferimenti

- R. Long, N. Sturtevant, M. Buro, M. Bowling — *Understanding the Success of Perfect Information
  Monte Carlo Sampling in Game Tree Search* (AAAI 2010) — limiti PIMC: strategy fusion, non-locality.
- D. Rebstock, C. Solinas, M. Buro, N. Sturtevant — *Policy Based Inference in Trick-Taking Card
  Games* (CoG 2019, arXiv:1905.10911) — determinizzazioni pesate da un modello dell'avversario
  migliorano il SOTA Skat.
- D. Rebstock, C. Solinas, M. Buro — *Learning Policies from Human Data for Skat*
  (arXiv:1905.10907).
- T. Anthony, Z. Tian, D. Barber — *Thinking Fast and Slow with Deep Learning and Tree Search*
  (Expert Iteration, NIPS 2017) — il loop policy↔search iterato vs distillazione one-shot.
- D. Zha et al. — *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*
  (ICML 2021, arXiv:2106.06135) — Deep MC + storia encodata, from scratch, gioco di carte.
- J. Cotarelo et al. — *Perfect Information Monte Carlo with Postponing Reasoning*
  (EPIMC, arXiv:2408.02380) — mitigazione strategy fusion.

## 11. Sequenza operativa consigliata

1. Fase 0 (a+b+c in parallelo, notti CPU) → fissa target e priorità.
2. Fase 1 encoder v4 → gate 4.3.
3. Fase 2 belief + determinizzazioni pesate → primo artefatto promuovibile
   (`bc_model_pimc_belief_*` come nuovo campione runtime).
4. Fase 3 ExIt → l'obiettivo primario.
5. Fase 4 architettura dentro ExIt, solo se/quando il loop satura per capacità.

Ogni promozione segue i criteri standard di PLAN.md (seat-fair con CI su coppie, holdout non
peggiore, decision-quality, report modelli).
