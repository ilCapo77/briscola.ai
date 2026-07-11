# Piano di ricerca: prossima iterazione del modello

> Stato: proposta operativa, 2026-07-11.
> Questo documento approfondisce le sette piste sintetizzate in `PLAN.md`; non autorizza
> automaticamente training lunghi né promozioni. `PLAN.md` resta la fonte di verità su
> quale fase eseguire adesso.

## 1. Obiettivo e baseline

La baseline da preservare è:

- policy `best_a2c_v13.npz`, encoder v4, `369 -> 256 -> 40`, senza guard runtime;
- default prodotto `bc_model_pimc_belief_16x8` con belief v0 e solver finale;
- valutazioni seat-fair paired, stessi mazzi e posti scambiati;
- anti-cheat invariato: policy, belief, reward e search ricevono solo informazione lecita.

Il plateau v11-v13 non prova da solo che la rete sia troppo piccola. Può dipendere da
capacità, rumore del trainer, simmetrie non apprese, qualità della belief o dal fatto che
parte del vantaggio della search non sia comprimibile in una policy reattiva. L'ordine
del piano serve a distinguere queste cause prima di pagare un'altra run lunga.

## 2. Regole comuni degli esperimenti

Ogni ablation deve cambiare **una sola variabile** e registrare:

- commit, hash SHA-256 degli asset, encoder e metadati completi;
- seed di training, seed suite di evaluation, roster e configurazione dei wrapper;
- giochi, tempo totale, CPU media e p95 quando cambia il runtime;
- gate policy-only, default PIMC 16x8 e decision quality;
- CI paired e risultati di almeno tre training seed per una promozione del trainer.

Una CI di evaluation stretta misura l'incertezza delle partite di **quel checkpoint**,
non la variabilità del training. Per questo un singolo candidato fortunato non basta.

### Criterio di promozione

Una variante può essere promossa solo se:

1. non regredisce materialmente nei gate policy-only e PIMC 16x8;
2. non peggiora spreco/overkill o altri comportamenti già corretti;
3. il vantaggio si ripete su più seed di training, oppure l'intervento è una modifica
   runtime deterministica verificata sugli stessi campioni;
4. costo e latenza sono compatibili con il default web;
5. nessun test o nuova API introduce accesso a mani o mazzo nascosti.

## 3. Pista 1: salute e capacità della MLP

### Domanda

La hidden layer da 256 neuroni usa davvero la propria capacità, oppure contiene neuroni
quasi inattivi o contributi duplicati che indicano un problema di ottimizzazione?

### Stato del codice

Sono già disponibili i pesi `w1/b1/w2/b2` in
`src/briscola_ai/ai/models/bc_model.py`, l'encoder v4 e raccolte di osservazioni lecite
simili a quelle di `scripts/style_feature_probe.py`. `scripts/widen_mlp_net2net.py`
implementa invece Net2Net per copia: è una tecnica diversa dalla nuova ablation
zero-output e preserva esattamente la funzione solo con rumore zero.

### Da implementare

Una sonda riproducibile su suite fissa che, per fase della partita, misuri:

- frequenza con cui ogni pre-attivazione è positiva;
- distribuzione e magnitudine delle attivazioni;
- contributo del neurone alle sole azioni legali, rimuovendo la componente costante che
  non può cambiare la scelta;
- cambi dell'argmax quando il neurone viene ablatto;
- collegamento ai gradienti actor, critic e trunk durante piccoli smoke training.

Un neurone mai attivo nella suite va chiamato **inattivo sulla suite**, non morto in
assoluto. Anche 256 neuroni attivi non dimostrano che serva più capacità: la prova causale
resta un training controllato.

### Ablation di capacità

Se la diagnosi la giustifica, confrontare v13 con estensioni `256 -> 320` e `256 -> 384`.
I nuovi ingressi devono avere inizializzazione random/He, mentre le nuove righe verso
actor e critic devono partire da zero. Così la funzione iniziale è identica, ma i nuovi
neuroni possono attivarsi e ricevere gradiente. Azzerare sia ingressi sia uscite creerebbe
ReLU permanentemente inattive.

Test richiesti:

- logits e valore iniziali identici a v13 su una suite di osservazioni;
- nuovi neuroni effettivamente attivi;
- pesi uscenti nuovi aggiornati dopo almeno un optimizer step;
- save/load e metadati coerenti;
- confronto distinto con Net2Net `noise=0`, senza mescolare le due tecniche.

### Decisione

**GO** solo se widening zero-output migliora in modo ripetibile i gate su tre seed.
**STOP** se attivazioni e ablation non mostrano un collo di bottiglia oppure il widening
aumenta varianza senza vantaggio ripetibile.

## 4. Pista 2: simmetria alla rinomina dei semi

### Domanda

Il modello cambia decisione quando denari, coppe, spade e bastoni vengono soltanto
rinominati, rinominando coerentemente anche briscola, tavolo e storia?

### Da implementare

Serve un helper canonico che permuti un'intera `PlayerObservation`:

- mano, briscola, tavolo e `trick_history`;
- `seen_cards_onehot` e `out_of_play_cards_onehot`;
- action mask e action id, usando la convenzione `suit * 10 + rank`;
- output del modello, rimappato poi nei semi originali prima del confronto.

La suite deve provare tutte le 24 permutazioni. L'identità è un controllo; agreement,
flip dell'argmax e divergenza Jensen-Shannon vanno calcolati sulle altre 23.

Test richiesti:

- round trip permutazione/inversa senza perdita;
- carte legali e feature coerenti dopo il remapping;
- agente sintetico equivariant con agreement 100% e divergenza zero;
- nessun uso di `GameState` completo.

### Possibili interventi

In ordine di costo:

1. data augmentation con permutazioni casuali durante il training;
2. loss di consistenza tra osservazione originale e permutata;
3. media delle 24 predizioni a inference, solo come upper bound perché costosa;
4. architettura esplicitamente equivariant, valutata soltanto nella pista 7.

Embedding assoluti separati per i quattro semi **non** garantiscono la simmetria. Una
futura architettura deve usare ruoli relativi alla briscola/lead oppure condivisione di
pesi equivariant.

### Decisione

**GO** verso augmentation/consistency se la sonda trova asimmetrie ripetibili e
l'intervento riduce i flip senza regressione di forza. **STOP** se v13 è già quasi
equivariant o se la regolarizzazione rende le probabilità più simmetriche ma gioca peggio.

## 5. Pista 3: dose e budget adattivo PIMC

### Domanda

Quanta forza aggiungono davvero 32 o 64 determinizzazioni nella finestra 8, e possiamo
spendere il budget alto soltanto sulle decisioni incerte?

### Stato del codice

`PIMCAgent`, `PIMCSearchDiagnostics` e `scripts/evaluate_pimc.py` forniscono la base.
Oggi manca la latenza per singola decisione, quindi non abbiamo p50/p95, e l'harness non
confronta ancora in modo simmetrico due configurazioni belief differenti: il belief
passato dalla CLI viene applicato soltanto al lato A.

### Fase fissa 16/32/64

Prima estendere l'harness affinché i due lati condividano:

- policy v13, belief, `uniform_mix`, finestra 8 e solver;
- stessi seed e stesso prefisso di campioni per le prime 16/32 determinizzazioni;
- diagnostica per decisione e latenza p50/p95.

Il confronto principale è forza a pari costo CPU, non semplice agreement con 64.

### Budget adattivo

Solo se 32/64 mostrano headroom, introdurre campionamento a blocchi con minimo e massimo.
Lo stop deve considerare stabilità del primo e secondo candidato, non soltanto il margine
medio dell'ultimo blocco. Controllare più volte una CI nominale al 95% introduce bias da
optional stopping: una prima versione va dichiarata euristica e validata direttamente
contro fixed-64; una regola sequenziale formale è un esperimento successivo.

Test richiesti:

- budget sempre nei limiti e modalità disattivata identica al fixed budget;
- nessuno stop su tie o sequenze sintetiche ambigue;
- stop anticipato su sequenze chiaramente separate;
- determinizzazioni effettive e latenza sempre registrate;
- anti-cheat invariato: tutti i campioni partono dalla sola osservazione.

### Decisione

**GO** al budget adattivo se esiste vantaggio 32/64 e la variante mantiene la forza con
CPU media/p95 inferiori. **STOP** se la curva 16/32/64 è piatta; non riaprire la finestra
10, già chiusa negativa.

## 6. Pista 4: rendere A2C meno rumoroso

### Stato reale del trainer

`scripts/train_a2c.py` usa un actor-critic con return Monte Carlo completi e trunk
condiviso. Nei warm-start il critic `wv/bv` viene oggi sempre reinizializzato a zero.
Non esistono ancora normalizzazione degli advantage né gradient clipping globale.

### Prima fase: osservabilità

Prima di cambiare l'algoritmo registrare:

- media/std degli advantage per optimizer update;
- explained variance e loss del critic;
- norme dei gradienti separate per actor, critic e trunk;
- activation rate della hidden layer;
- rapporto tra aggiornamento e norma dei parametri.

### Ablation, una alla volta

1. critic `reset` attuale contro `reuse` dal checkpoint;
2. normalizzazione advantage sull'intero optimizer update, non per singola partita;
3. global gradient clipping immediatamente prima di Adam;
4. learning-rate decay solo dopo aver isolato i tre punti precedenti;
5. stop-gradient o trunk separato actor/critic soltanto se la diagnostica mostra
   interferenza, perché cambia più profondamente capacità e costo.

I path per-step e Numba batch devono applicare la stessa definizione. Servono test su
gradienti sintetici, caricamento reale del critic, metadati e parità dei due accumulatori.

### Decisione

Ogni variante passa prima uno screening breve su tre seed. **GO** a una run lunga solo
per un intervento con varianza ridotta e mediana non regressiva; **STOP** se il beneficio
appare in un solo seed o richiede combinare più modifiche non interpretabili.

PPO non è il passo successivo automatico: prima va stabilito se il collo di bottiglia è
nel trainer attuale.

## 7. Pista 5: training davvero paired

### Problema

Il flag attuale `train_a2c.py --seat-fair` alterna il posto della policy, ma genera un
mazzo diverso per ogni partita. Non è quindi paired come l'evaluation.

### Da implementare

Una schedule pura e riproducibile `(game_seed, policy_seat, opponent)` in cui ogni coppia:

- usa lo stesso seed e lo stesso opponent campionato dal mix;
- gioca una volta per seat;
- viene consumata interamente prima dello stesso optimizer update.

`num_games` deve essere pari. Anche `update_every` deve essere pari oppure il buffer deve
garantire che una coppia non attraversi due versioni diverse della policy. I collector
Numba accettano già array `game_seeds` e `policy_seats`; la schedule va resa identica nei
path domain, fast Python e Numba.

### Disegno sperimentale

Il pairing dimezza i mazzi distinti a parità di partite. Confrontare quindi:

1. stesso numero totale di partite/optimizer update;
2. stesso numero di mazzi distinti, accettando il doppio delle partite paired;
3. almeno tre seed indipendenti di training per ciascun regime.

Metriche: varianza tra run, velocità di apprendimento, gate finali e costo. Il pairing può
ridurre bias e rumore, ma non è garantito che migliori il modello.

Test richiesti: seed/opponent uguali e seat `{0,1}` in ogni coppia, schedule deterministica,
nessuna coppia spezzata da un update e rifiuto esplicito di configurazioni dispari.

## 8. Pista 6: belief v1 multi-stile

### Problema

La belief ufficiale `belief_v0_h128_50k_seed20260702.npz` è stata addestrata su 50k
osservazioni mirror di v7. `generate_belief_dataset.py` usa oggi la stessa policy su
entrambi i lati; non misura generalizzazione a stili diversi.

### Da implementare

- roster esplicito: v13, anchor nominata v11 ed euristiche con stili differenti;
- `opponent_id` per record come solo metadato di split, mai input della rete;
- split per partita e fold leave-one-opponent-out;
- metriche BCE/top-k già esistenti più Brier score e calibrazione/ECE;
- harness A/B simmetrico belief v0-v1 nello stesso PIMC 16x8.

Il fold va assegnato in base allo stile della mano avversaria che la belief cerca di
inferire, non al generico nome del matchup. Il full-state è lecito solo per costruire la
label `y`; input `x` e inferenza restano osservazioni parziali.

La sigmoid per-carta non impone l'esatta cardinalità della mano, quindi una metrica
offline migliore non basta. Il gate decisivo mantiene identici policy v13, D=16,
finestra 8, solver e uniform mix, cambiando soltanto belief v0/v1.

### Decisione

**GO** solo se v1 migliora calibrazione sui fold esclusi e poi batte/non regredisce v0
nel PIMC paired. **STOP** se migliora BCE/top-k ma non le decisioni della search.

## 9. Pista 7: nuova architettura o Q Monte Carlo

Questa fase parte solo se le piste precedenti dimostrano un limite di rappresentazione.
Non va combinata con modifiche al trainer o alla belief nello stesso esperimento.

### Opzione A: ramo residuale zero-output

È l'estensione minima della MLP e riusa la logica della pista 1. Deve partire con funzione
identica a v13, avere formato `.npz` esplicito e restare compatibile con inference NumPy.

### Opzione B: carte e storia con pesi condivisi

Usare uno scorer condiviso per carta e ruoli relativi a briscola/lead, più un piccolo
encoder sequenziale della sola `trick_history` pubblica. Embedding assoluti dei quattro
semi non bastano a garantire equivarianza. La storia non deve contenere ordine del mazzo
né mani nascoste.

### Opzione C: `Q(observation, card)` / Deep Monte Carlo

Una vera Q condivisa assegna un valore con lo stesso scorer a ogni carta legale. Può
chiamarsi Deep Monte Carlo solo se i target sono ritorni Monte Carlo stato-azione con
esplorazione sufficiente. Allenarla soltanto sulle azioni scelte crea selection bias;
usare action-values PIMC è invece distillazione del teacher, un esperimento diverso.

### Costo architetturale

Il repository non usa PyTorch/JAX e i kernel Numba assumono una MLP flat a una hidden
layer. Prima di implementare va quindi deciso:

- trainer manuale NumPy oppure dipendenza solo-training;
- formato export NumPy stabile per il runtime web;
- loader/catalogo, inferenza batch/singola e benchmark CPU;
- strategia fast/Numba senza portare framework pesanti nel processo web.

Test minimi: mask delle azioni illegali, save/load identico, equivarianza dei semi,
parità della storia pubblica, assenza di full-state e gate separati policy/PIMC/qualità.

### Decisione

Iniziare dal ramo residuale, che ha il minor numero di variabili. Embedding/history e Q
Monte Carlo sono due programmi distinti. **STOP** se il costo di tooling domina il
segnale oppure se il widening controllato non mostra alcun limite di capacità.

## 10. Sequenza operativa e criteri di stop

| Ordine | Fase | Costo iniziale | Output richiesto | Stop immediato |
|---:|---|---|---|---|
| 1 | Sonda salute MLP | basso | JSON riproducibile per fase/neurone | nessun segnale di capacità |
| 2 | Sonda simmetria semi | basso | agreement, JS, flip su 24 permutazioni | v13 già quasi equivariant |
| 3 | Dose PIMC 16/32/64 | medio | forza + CPU p50/p95 | curva piatta |
| 4 | Strumentazione/A2C | medio | gradienti, critic, tre ablation separate | segnale non ripetibile |
| 5 | Training paired | medio | schedule + confronto multi-seed | varianza/forza non migliori |
| 6 | Belief v1 | medio-alto | fold multi-stile + A/B PIMC | solo metriche offline migliori |
| 7 | Nuova architettura | alto | prototipo esportabile e gate completi | tooling/costo senza headroom |

Non avviare la fase successiva per inerzia. Ogni fase deve produrre un artefatto piccolo
e versionabile (JSON di sonda, manifest o nota tecnica) che consenta a una sessione futura
di ricostruire la decisione senza dipendere dalla memoria della conversazione.

## 11. Comandi già validi e lavoro mancante

Baseline riproducibili già disponibili:

```bash
uv run python scripts/behavior_profile.py \
  --model data/models/best_a2c_v13.npz \
  --opponents heuristic_trump_saver,mirror,heuristic_v1 \
  --num-games 2000

uv run python scripts/evaluate_pimc.py \
  --model data/models/best_a2c_v13.npz \
  --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --determinizations 16 \
  --max-unknown-cards 8 \
  --num-games 2000
```

Il secondo comando è solo una baseline: prima del confronto belief 16/32/64 l'harness
deve diventare simmetrico e registrare le latenze per decisione.

Non esistono ancora comandi affidabili per sonda MLP, permutazioni dei semi, critic
reuse/normalizzazione/clipping, schedule paired, roster belief v1 o nuove architetture.
I relativi flag e script vanno implementati e testati prima di preparare ricette lunghe.
