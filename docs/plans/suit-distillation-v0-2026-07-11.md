# Distillazione del teacher simmetrico v0 (2026-07-11)

> Screening da 10.000 partite superato, nessuna promozione. Evidenza:
> [suit_distillation_v0.v1.json](../reports/evidence/suit_distillation_v0.v1.json).

## Domanda e verdetto

La media dei logits v13 sulle 24 rinomine gioca meglio della policy originale ma richiede
un batch 24x. Possiamo trasferire quel comportamento in una sola MLP v4, mantenendo un
forward normale a inference?

**Verdetto screening: sì.** Il candidato distillato su 10.000 partite:

- porta l'accordo col teacher sull'holdout da `86,83%` a **`92,88%`**;
- riduce il flip dei semi da `18,19%` di v13 a **`10,23%`**, superando il gate `<12%`;
- batte direttamente v13 di **`+0,51` punti/partita**, CI95 `+0,11..+0,92`;
- resta neutro contro il teacher 24x: `-0,23`, CI95 `-0,59..+0,13`;
- conserva il miglioramento comportamentale: overkill su piatto povero `5,5%`, fra v13
  (`8,0%`) e teacher (`3,9%`).

**GO al corpus indipendente da 50.000 partite; nessuna promozione runtime o PIMC prima di
quel gate.**

## Pipeline implementata

La distillazione non riusa il JSONL BC: 380.000 osservazioni v4 complete sarebbero molto
più grandi e lente da parsare. Il nuovo formato numerico compresso contiene:

- feature `(N, 369)` float32 e action mask `(N, 40)`;
- probabilità soft del teacher e relativo argmax;
- `game_id` e split per ogni esempio;
- metadati essenziali su modello, hash, seed, roster e temperatura.

Ogni partita produce 38 esempi: le due mosse finali con una sola carta vengono escluse
perché la softmax ha gradiente nullo. Il teacher riceve soltanto `PlayerObservation`.

### Split senza leakage

Le 10.000 partite sono assegnate deterministicamente prima della raccolta:

| split | partite | esempi |
|---|---:|---:|
| train | 8.000 | 304.000 |
| validation | 1.000 | 38.000 |
| test | 1.000 | 38.000 |

Tutte le decisioni di una partita restano nello stesso split. Il test non partecipa alla
selezione del checkpoint: viene letto sul modello iniziale e poi sul migliore per KL di
validation. Questo chiude il rischio di leakage per la nuova pipeline; `train_bc.py` e il
value trainer generico conservano il debito storico dello split per record.

### Distribuzione delle traiettorie

V13 occupa un posto alternato; l'altro agente viene campionato dal roster preregistrato:

| avversario | peso | partite osservate |
|---|---:|---:|
| mirror v13 | 50% | 4.905 |
| heuristic_trump_saver | 20% | 2.040 |
| heuristic_v1 | 15% | 1.572 |
| heuristic_v2 | 10% | 998 |
| random | 5% | 485 |

Ogni stato visitato viene etichettato dal teacher simmetrico, anche quando la traiettoria
è stata prodotta dall'euristica. Il corpus copre quindi stili differenti senza introdurre
informazione nascosta.

## Training

Il candidato parte esattamente da v13 (`369 -> 256 -> 40`) e usa Adam, 5 epoche, batch
1024, learning rate `2e-4`, weight decay `1e-6`. Per ogni minibatch viene affiancata una
copia supervisionata con una delle 23 rinomine non identità: feature, mask, probabilità e
argmax vengono trasformati insieme.

Questa copia è valida dove la paired augmentation RL non lo era: il target soft deriva
direttamente dal teacher sulla posizione, non da un'azione campionata da un'altra policy.

| checkpoint | val KL | val agreement |
|---:|---:|---:|
| v13 iniziale | 0,62658 | 86,69% |
| epoca 1 | 0,34103 | 90,01% |
| epoca 3 | 0,21874 | 91,63% |
| **epoca 5** | **0,17078** | **92,50%** |

Sul test indipendente: KL `0,16861`, agreement `92,88%`. Il controllo senza augmentation
si ferma a KL `0,19635` e agreement `92,29%`. Nel direct match paired-vs-noaug la forza è
neutra (`+0,08`, CI `-0,11..+0,28`): la copia migliora imitazione e simmetria, ma questo
screening non attribuisce ad essa un vantaggio di gioco separato.

## Simmetria

Sonda canonica su 4.096 osservazioni e 94.208 rinomine non identità:

| modello | flip | JS media | stati con almeno un flip |
|---|---:|---:|---:|
| v13 | 18,19% | 0,14124 bit | 51,17% |
| distillato senza augmentation | 10,95% | 0,06773 bit | n/d |
| **distillato paired** | **10,23%** | **0,06143 bit** | **31,37%** |
| teacher 24x | 0% | 0 bit | 0% |

La MLP ordinaria non eredita la garanzia matematica del teacher, ma supera il gate `<12%`
senza comprimere il margine: gap top-2 medio `0,909`, vicino a v13 (`~0,93`).

## Forza policy-only

Tutte le righe usano la seed suite medium, 10.000 partite seat-fair e CI sulle coppie.

| agente A | agente B | punti A-B | CI95 |
|---|---|---:|---:|
| **distillato paired** | v13 | **+0,51** | **+0,11..+0,92** |
| distillato paired | teacher 24x | -0,23 | -0,59..+0,13 |
| distillato paired | no augmentation | +0,08 | -0,11..+0,28 |
| distillato paired | heuristic_v1 | +22,01 | +21,47..+22,55 |
| distillato paired | heuristic_trump_saver | +15,57 | +15,05..+16,09 |

Il candidato recupera circa il 57% del vantaggio diretto teacher-vs-v13 (`0,51/0,90`),
ma la stima è solo descrittiva: i due direct match condividono la suite ma non costituiscono
una singola differenza statistica paired a tre agenti.

## Qualità decisionale

Gate domain da 10.000 partite contro `heuristic_v1`:

| metrica | v13 | distillato | teacher 24x |
|---|---:|---:|---:|
| overkill complessivo | 22,05% | **21,86%** | 22,05% |
| overkill su piatto povero | 8,01% | **5,51%** | 3,91% |
| trump waste | 0,120% | **0,080%** | 0,050% |

Il candidato resta intermedio fra allievo iniziale e teacher anche nei comportamenti, senza
riaprire il difetto corretto da v13.

## Costi e artefatti locali

- Generazione 10k: `70,54 s` + `5,69 s` di compressione, 142 partite/s.
- Dataset compresso: `32.547.075` byte; SHA-256
  `cdcfb37ec20b32deb76fa785e118c1083a60d6e7cc8dba474787eec6ec443275`.
- Training paired: `6,01 s`.
- Modello: `424.740` byte; SHA-256
  `a837b3b9994a307bf58403797d4c72c7f038678c47c8ec2eef2fc8cb12c4cce5`.

Dataset, modelli candidati e report grezzi restano gitignored. L'evidenza sintetica
versionata contiene i numeri necessari a riprodurre la decisione.

## Prossimo gate

Generare un corpus indipendente da 50.000 partite con seed `20260712`, stesso roster e
stesso split 80/10/10. Produce 1,9 milioni di esempi; il formato monolitico richiede circa
3,2 GB di RAM durante la raccolta e circa 160 MB su disco. Per iterazioni successive sarà
preferibile introdurre shard streaming, ma non è un prerequisito per questo gate singolo.

Il run 50k passa soltanto se:

1. agreement test e KL migliorano rispetto al 10k;
2. flip resta sotto 10,23% o almeno non supera il gate 12%;
3. forza diretta contro v13 resta positiva e contro teacher non peggiora;
4. overkill povero non supera v13;
5. solo dopo, il candidato entra nel confronto PIMC 16x8 a pari configurazione.

Comando operativo in `PLAN.md`; il job di raccolta supera cinque minuti e va lanciato dal
maintainer con `nohup`.
