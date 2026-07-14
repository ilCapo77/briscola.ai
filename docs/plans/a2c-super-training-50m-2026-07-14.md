# Super training A2C seriale da 50 milioni di partite

Data: 2026-07-14  
Stato: **trainer validato; primo blocco da 10M non ancora avviato**  
Modello live invariato: `best_a2c_v14.npz`

## Decisione

Il costo monetario del calcolo e' nullo. Questo rende ragionevole un ultimo test
controllato dell'ipotesi piu' semplice rimasta:

> Una continuazione A2C molto piu' lunga, senza cambiare architettura o ricetta, riesce
> ad accumulare piccoli miglioramenti strategici che le sonde locali non isolano?

Il run non presume che la risposta sia positiva. Serve a distinguere un vero plateau
dalla possibilita' che v14 abbia soltanto ricevuto troppo poco reinforcement learning
dopo la distillazione. V14 nasce infatti da v13 tramite training supervisionato e non ha
mai svolto una continuazione A2C lunga; in passato la scala 20M/30M ha prodotto progressi
fra v8, v9 e v10.

Il primo run e' uno **scouting**, non v15. Un singolo seed non puo' essere promosso.

## Una sola variabile

La variabile intenzionale e' il numero di partite. Tutto il resto resta congelato sulla
ricetta seriale gia' verificata con v14:

| componente | valore congelato |
|---|---|
| init | `data/models/best_a2c_v14.npz` |
| BC anchor | v14, beta `0,01` |
| encoder / hidden | v4, 256 |
| training schedule | `serial`, seat alternata |
| rollout | fast, batch Numba |
| partite | `50.000.000` |
| seed scouting | `20260723` |
| update | 20 partite |
| learning rate | `0,0003` |
| entropy beta | `0,0005` |
| value coefficient / gamma | `0,5` / `1,0` |
| shaping | overkill `gap`, beta `0,3`, piatto massimo 2 punti |
| guard runtime | disabilitato |
| loss sui semi | nessuna augmentation/KL/margin durante A2C |

### Avversari

Il modello non gioca soltanto contro la policy v14 pura. Il mix e' quello gia' usato
nella linea v13/v14:

| avversario | quota | base |
|---|---:|---|
| `bc_model` | 15% | v14 diretta |
| `bc_model_pimc_belief` | 40% | v14 + belief v0, PIMC 16x8 |
| `bc_model_value_lookahead_8x8` | 20% | v14 + value lookahead |
| `heuristic_trump_saver` | 12% | stile conservativo |
| `heuristic_v1` | 4% | euristica |
| `heuristic_v2` | 6% | euristica |
| `random` | 3% | diversita' minima |

Il 75% usa quindi v14 come policy di base con tre livelli decisionali diversi; il 25%
evita che il candidato diventi soltanto uno specialista anti-v14. Policy, belief, value
e roster avversari restano congelati per tutte le 50M partite.

## Fase 0 - Trainer long-run completato

L'audit iniziale ha trovato tre strutture che crescevano con la lunghezza del training e
checkpoint che non consentivano una ripresa esatta. La fase 0 le ha corrette prima di
autorizzare il primo blocco.

### 0.1 Schedule a memoria costante

`train_a2c.py` costruisce oggi una tuple con un oggetto per ogni partita. A 50M la sola
schedule richiederebbe molti gigabyte prima del primo rollout.

Il trainer ora:

- generare seed, seat e avversario a flusso, al massimo un optimizer batch alla volta;
- mantenere incrementalmente lo SHA-256 della schedule consumata;
- produrre ai checkpoint il digest del prefisso senza copiare o riattraversare la storia;
- preservare bit-per-bit il training seriale esistente sugli stessi seed.

### 0.2 Metriche realmente compatte

`--metrics-mode summary` salva solo prima e ultima riga nel file, ma oggi conserva
comunque tutte le righe in RAM. Con update da 20 partite sarebbero 2,5 milioni di oggetti.

La modalita' summary mantiene soltanto:

- conteggio;
- prima e ultima riga;
- aggregati online eventualmente necessari.

La modalita' `full` deve restare invariata per compatibilita' sui run piccoli.

### 0.3 Checkpoint riprendibile e atomico

Il checkpoint corrente contiene pesi actor/critic, ma non momenti Adam, contatore update,
stato degli RNG e cursore/digest della schedule. Usarlo come nuovo `--init` cambierebbe il
training e azzererebbe di nuovo il critic.

Il resume state incorporato conserva:

- tutti i tensori `m/v` di Adam e il contatore `t`;
- actor, critic e metadati di configurazione;
- stato degli RNG e posizione della schedule streaming;
- digest del prefisso gia' consumato;
- hash degli asset congelati e fingerprint dei flag.

La scrittura deve essere atomica. Un test interrotto a un checkpoint e ripreso deve
produrre pesi, metriche e digest bit-identici al run senza interruzione.

### 0.4 Telemetria campionata

La diagnostica per ogni update produrrebbe 2,5 milioni di record. Il run lungo usera' una
frequenza esplicita, indicativamente una riga ogni 1.000 update, includendo sempre primo,
checkpoint e ultimo update. La telemetria resta passiva e non consuma RNG.

Il log testuale avra' un heartbeat circa ogni 20.000 partite. Non salveremo osservazioni,
mani o carte nascoste.

### Gate di prontezza

Il training 50M era autorizzabile soltanto dopo:

1. test di parita' seriale vecchio/streaming;
2. test di resume bit-identico;
3. prova che summary e diagnostica campionata hanno memoria limitata;
4. smoke end-to-end con checkpoint e resume;
5. quality gate completo del repository.

### Ricevuta di prontezza

Tutti i gate sono superati:

- `TrainingGameScheduleStream` produce un batch alla volta e conserva soltanto RNG,
  contatore e una hash-chain SHA-256 estendibile dal checkpoint;
- `StreamingHistory` conserva tutte le righe in modalita' `full`, ma in `summary` mantiene
  conteggio, prima e ultima riga anche dopo 100.000 append nel test;
- i checkpoint `.npz` sono pubblicati atomicamente e contengono actor, critic, tutti i
  momenti Adam, update counter, RNG NumPy/Python, schedule, metriche e configurazione;
- il fingerprint rifiuta modifiche a iperparametri, asset o commit fra due segmenti;
- `--diagnostics-every` conserva primo, ultimo, checkpoint e un update ogni N, senza
  cambiare gradienti o RNG;
- il test end-to-end Numba continuo 8 partite contro 4+4 riprese confronta i sei tensori
  bit per bit, oltre a digest, metriche e diagnostica;
- lo smoke con la ricetta reale v14 (PIMC belief 16, value lookahead, anchor e roster
  completo) produce gli stessi tensori nel confronto 40 continuo contro 20+20;
- quality gate: ruff, mypy, docs-check e **671 test** passati.

## Fase 1 - Scouting in cinque blocchi da 10M

Il training e' diviso in blocchi riprendibili da 10 milioni di partite. Dopo ogni blocco
il processo si ferma, eseguiamo lo screen definito nella fase 2 e decidiamo se avviare il
blocco successivo. Il resume deve conservare esattamente Adam, critic, RNG e schedule:
cinque blocchi devono essere bit-identici a un run continuo con la stessa configurazione.

### Checkpoint tecnici e strategici

Checkpoint tecnici di sicurezza:

`5M, 15M, 25M, 35M, 45M`.

Servono soltanto a recuperare la seconda meta' di un blocco dopo un crash. Non vengono
valutati strategicamente e non sono eleggibili per la selezione del modello.

Checkpoint strategici e pause decisionali:

`10M, 20M, 30M, 40M, 50M`.

Questa granularita' e' congelata prima del run. Riduce le occasioni di selezionare un
picco fortunato e lascia circa sette ore di apprendimento fra due valutazioni, sulla base
del throughput misurato oggi.

### Artefatti

Directory prevista:

```text
benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/
  manifest.json
  train.log
  diagnostics.sampled.json
  models/
  resume/
  validation/
  final_gate/
```

Ogni ricevuta deve contenere commit, versione, seed, comando, hash di v14/belief/value,
configurazione, durata, checkpoint e digest della schedule.

### Tempo e spazio

I probe correnti impiegano circa 52,4 secondi per 20.000 partite con la stessa famiglia
di avversari. La proiezione lineare e' circa **7,3 ore per blocco** e **36 ore totali**;
il budget operativo completo va dichiarato come 36-48 ore e aggiornato dopo lo smoke.
Con metriche streaming, i modelli e gli stati di resume richiedono decine di megabyte,
non gigabyte.

### Comando del primo blocco 0-10M

Questo comando dichiara fin dall'inizio l'orizzonte totale da 50M, ma ferma il processo a
10M. I file a 5M e 10M contengono lo stato completo di resume; la riga diagnostica viene
conservata ogni 1.000 update, mentre il log stampa un heartbeat ogni 20.000 partite.

```bash
mkdir -p \
  benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/models \
  benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/resume \
  benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/validation \
  benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/final_gate

PYTHONUNBUFFERED=1 nohup caffeinate -dimsu \
  uv run python scripts/train_a2c.py \
  --init data/models/best_a2c_v14.npz \
  --out benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/models/a2c_v14_scale50m_seed20260723_at10m.npz \
  --encoder-version v4 \
  --rollout-engine fast --fast-rollout numba \
  --training-schedule serial --seat-fair \
  --opponent-mix bc_model:0.15,bc_model_pimc_belief:0.40,bc_model_value_lookahead_8x8:0.20,heuristic_trump_saver:0.12,heuristic_v1:0.04,heuristic_v2:0.06,random:0.03 \
  --opponent-model data/models/best_a2c_v14.npz \
  --opponent-belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --opponent-pimc-determinizations 16 \
  --opponent-value-model data/models/value_v1_v4_fullgame_h128_seed20260718.npz \
  --opponent-value-max-unknown-cards 8 \
  --bc-anchor data/models/best_a2c_v14.npz --bc-anchor-beta 0.01 \
  --overkill-penalty-mode gap --overkill-penalty-beta 0.3 \
  --overkill-low-lead-points-max 2 \
  --lr 0.0003 --weight-decay 0 \
  --entropy-beta 0.0005 --value-coef 0.5 --gamma 1.0 \
  --suit-augmentation off --suit-consistency-beta 0 --suit-margin-beta 0 \
  --update-every 20 --log-every 1000 \
  --metrics-mode summary \
  --diagnostics-json benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/diagnostics_0_10m.sampled.json \
  --diagnostics-every 1000 \
  --num-games 50000000 --stop-after-games 10000000 \
  --checkpoint-games 5000000,10000000 \
  --checkpoint-dir benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/resume \
  --checkpoint-prefix a2c_v14_scale50m_seed20260723 \
  --seed 20260723 \
  > benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/train_0_10m.log 2>&1 &

echo $! > benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/train_0_10m.pid
```

Controllo non invasivo:

```bash
tail -f benchmarks/experiments/a2c_v14_serial_scale50m_seed20260723/train_0_10m.log
```

Il blocco e' completo quando il log contiene `Saved model` e il processo non esiste piu'.
Gli artefatti attesi sono il checkpoint tecnico `..._5m.npz`, quello strategico
`..._10m.npz`, il modello `..._at10m.npz` e la diagnostica campionata. Non avviare il
blocco 10-20M prima dello screen della fase 2.

## Fase 2 - Screen ogni 10M senza usare il test finale

La suite di selezione e' congelata a 4.000 partite seat-fair con seed di coppia
consecutivi a partire da `5.000.000`. Viene applicata soltanto ai cinque checkpoint
strategici, durante le pause fra i blocchi.

Per ogni checkpoint misuriamo:

- policy-only contro v14 sugli stessi mazzi e posti scambiati;
- sonda dei semi sulle stesse celle avversario/fase;
- decision quality contro `heuristic_v1` senza guard.

Un checkpoint e' eleggibile per la successiva selezione soltanto se:

1. il vantaggio policy-only ha limite CI95 sopra zero, **oppure** il checkpoint corrente
   e quello strategico precedente hanno entrambi stima almeno `+0,20` punti;
2. il flip dell'argmax sotto rinomina dei semi non supera `8,0%` (v14: `6,04%`);
3. l'overkill sui piatti poveri non supera `6,0%` (v14: `4,17%`);
4. non usa guard runtime e resta compatibile con encoder/catalogo.

### Decisione fra i blocchi

- A 10M si continua normalmente fino a 20M se non esiste un hard failure: un solo blocco
  non basta a dichiarare inutile la scala.
- Da 20M in poi si continua se il checkpoint e' eleggibile oppure se il risultato di
  forza resta incerto e simmetria/stile sono sani.
- Si ferma per danno evidente: regressione di forza con limite superiore CI95 sotto
  zero, flip dei semi oltre `12%`, overkill povero oltre `8%`, numeri non finiti o stato
  di resume non verificabile.
- Se il checkpoint corrente e il precedente sono entrambi non positivi e nessun
  checkpoint precedente e' eleggibile, si puo' chiudere per futilita' senza consumare il
  blocco successivo.

Alla fine, o dopo uno stop preregistrato, scegliamo fra i soli checkpoint strategici
eleggibili quello con la differenza punti policy-only piu' alta. Se nessuno e'
eleggibile, il verdetto e' `STOP scaling`: non si cercano nuove soglie e non si promuove
l'ultimo file solo perche' ha completato piu' partite.

## Fase 3 - Gate finale sigillato

Il checkpoint scelto viene aperto una sola volta sulla suite finale, distinta dalla
selezione: 10.000 partite seat-fair con seed di coppia consecutivi da `6.000.000`.

Gate obbligatori:

1. **policy-only vs v14:** differenza punti positiva con limite inferiore CI95 `> 0`;
2. **default live vs v14:** PIMC belief 16x8 e solver identici, limite inferiore CI95
   `> 0` su 10.000 partite;
3. **simmetria/stile:** restano entro i tetti `8,0%` e `6,0%` usati in selezione;
4. **costo runtime:** una singola inferenza MLP, stesso formato e nessun nuovo asset.

Fallire un gate significa nessuna promozione. Un risultato compatibile con la parita'
non diventa positivo aumentando a posteriori il numero di partite.

## Fase 4 - Repliche prima di v15

Anche un successo completo del seed `20260723` autorizza soltanto due repliche. Si usa
esattamente l'orizzonte del checkpoint scelto, non necessariamente 50M, con seed nuovi e
stessa configurazione.

La ricetta e' replicata soltanto se:

- almeno due seed su tre hanno stima policy-only positiva contro v14;
- la mediana dei tre delta e' positiva;
- nessun seed mostra una sconfitta con limite superiore CI95 sotto zero;
- simmetria e stile non dipendono da un solo seed fortunato.

Solo allora esiste evidenza che la ricetta, e non un checkpoint fortunato, migliori v14.

## Fase 5 - Distillazione condizionata

La distillazione non e' automatica e non usa le vecchie etichette argmax PIMC, ramo gia'
chiuso. Per il candidato replicato si costruisce prima il teacher esatto sulle 24
rinomine dei semi. Il corpus finale usa **250.000 partite indipendenti**, cinque volte il
corpus con cui e' stata prodotta v14. Non si genera un corpus per ogni checkpoint da 10M:
la spesa e' autorizzata soltanto dopo selezione, gate finale e repliche della ricetta.

La proiezione lineare dall'esperimento v14 e' circa 30-35 minuti per raccolta, teacher 24x
e compressione. Il formato monolitico corrente non e' pero' adatto: 50.000 partite
occupavano circa 163 MB compressi ma 3,2 GB in RAM, quindi 250.000 arriverebbero a circa
814 MB su disco e 16 GB in memoria. Prima della fase 5 il dataset e il trainer devono
supportare shard deterministici, indicativamente 10 shard da 25.000 partite, letti uno
alla volta e con split sempre per partita. Seed, roster, assegnazione agli split e ordine
degli shard vanno registrati nel manifest.

- Se il candidato e' gia' simmetrico e il teacher non migliora il gioco, si conserva la
  policy grezza.
- Se il candidato e' piu' forte ma meno simmetrico, oppure il teacher a 24 viste lo
  migliora, si distillano i logits medi in una singola MLP.
- Il distillato deve battere o pareggiare il candidato grezzo e superare nuovamente v14
  sia policy-only sia nel PIMC 16x8.

La distillazione trasferisce e ripulisce la strategia trovata dal super training; non e'
considerata una fonte autonoma di nuova strategia.

## Esiti interpretabili

| esito | significato | decisione |
|---|---|---|
| nessun checkpoint 10M eleggibile | piu' training della stessa ricetta non basta | chiudere scaling |
| segnale validation, fallimento finale | selezione fortunata o fragile | chiudere senza riaprire soglie |
| raw forte, PIMC neutro | vantaggio assorbito dalla search live | nessuna promozione live |
| un seed forte, repliche incoerenti | varianza di training | nessuna v15 |
| tre seed coerenti, gate completi | nuova ricetta di forza | valutare distillazione e promozione |

Questo esperimento e' una verifica della scala, non la prova che 50M siano di per se'
meglio di 5M. Il numero sul filename non entra in alcun gate.
