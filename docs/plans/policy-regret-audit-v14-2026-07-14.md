# Audit automatico del regret di v14 (2026-07-14)

> Protocollo congelato prima del run formale. Esito: **in attesa della suite
> 192 osservazioni x 64 determinizzazioni**.

## Domanda

Le piste su simmetria, capacita', belief, dose PIMC e schedule A2C hanno escluso diverse
correzioni generiche. Prima di scegliere un'altra architettura o un nuovo training,
vogliamo quindi misurare in quali decisioni v14 lasci davvero punti a un'alternativa.

Perdere una partita non rende sbagliata una singola carta: puo' essere soltanto sfortuna.
La sonda congela invece l'informazione disponibile al giocatore, prova tutte le carte
legali negli stessi mondi plausibili e misura il vantaggio della migliore alternativa.

## Contratto anti-cheat

`estimate_policy_regret` accetta una `PlayerObservation`, mai il `GameState` reale. Usa:

- mano del decisore, tavolo, punteggi, dimensione del mazzo e storia pubblica;
- belief v0 per pesare le carte ignote nelle determinizzazioni;
- seed della partita soltanto come provenance del collector, mai come input dello
  stimatore;
- solver esatto a mazzo vuoto, dove la mano avversaria e' deducibile dall'osservazione.

Il report conserva il contesto pubblico separato dalla provenance. Non serializza mano
avversaria o ordine reale del mazzo.

## Suite bilanciata

La raccolta usa coppie seat-fair con stesso mazzo e posti scambiati. La policy v14 gioca
contro:

- mirror v14;
- `heuristic_trump_saver`;
- `heuristic_v1`.

Le decisioni forzate sono escluse. Le 192 osservazioni riempiono esattamente 24 celle:

`3 avversari x 4 fasi x 2 posizioni x 8 osservazioni`.

Le fasi sono `early`, `mid`, `pimc_window` (al massimo 8 carte vive ignote, mazzo non
vuoto) ed `endgame`; le posizioni sono apertura e risposta. Questo impedisce che un
avversario o una fase facile dominino il risultato aggregato.

## Stima controfattuale

Fuori dall'endgame, per ogni osservazione:

1. si campionano 64 stati nascosti compatibili con la sola osservazione, pesati da
   belief v0 con mix uniforme `0,10`;
2. in ogni stato si provano tutte le carte della mano;
3. tutte le alternative condividono stato determinizzato e seed di rollout;
4. v14 completa la partita per entrambi i lati, con solver esatto nel finale;
5. le prime 32 righe scelgono una sola alternativa candidata;
6. le ultime 32, mai usate per sceglierla, stimano il delta paired
   `alternativa - scelta v14`.

Lo split selection/evaluation evita di scegliere la carta fortunata e certificarla sugli
stessi campioni. A mazzo vuoto non servono campioni: tutte le azioni sono valutate con
minimax esatto.

## Definizione di errore affidabile

Una decisione campionaria e' affidabile soltanto se:

- l'alternativa scelta sul primo split e' diversa dalla carta v14;
- il regret medio sul secondo split e' almeno `1,0` punto;
- il limite inferiore dell'intervallo normale con `z=2,576` (circa 99%) e' maggiore di
  zero.

Il 99%, piu' severo del 95% usato nelle valutazioni aggregate, riduce i falsi allarmi
perche' qui controlliamo molte decisioni separatamente. Nell'endgame esatto basta un
regret di almeno `1,0` punto.

## Tassonomia automatica

Le etichette descrivono soltanto il cambio di carta, senza pretendere di leggere la
causa interna della rete:

- mancata risposta vincente o mancata cattura di una presa ricca;
- briscola anticipata, passaggio da briscola a non-briscola o overkill;
- esposizione/scarto di un carico;
- rinuncia intenzionale alla presa corrente, anche quando vale pochi punti;
- `other` quando nessuna regola pubblica descrive il caso.

Ogni decisione e' inoltre distinta per layer del prodotto:

- `policy_only`: fasi early/mid, dove la scelta v14 arriva direttamente al gioco;
- `pimc_search_window`: il default live esegue gia' la search;
- `exact_solver`: il default live sostituisce gia' la policy col solver.

Gli ultimi due gruppi restano diagnostici, ma non possono da soli giustificare un nuovo
training per migliorare il comportamento live.

## Gate di instradamento

La sonda non promuove modelli. Autorizza la progettazione di **un solo intervento
mirato** soltanto se una stessa etichetta diversa da `other`, nel gruppo `policy_only`:

1. compare in almeno 3 errori affidabili;
2. attraversa almeno 2 avversari;
3. attraversa almeno 2 coppie di partite.

Il report assegna automaticamente uno dei verdetti:

- `actionable_policy_error_cluster`: cluster completo, si puo' progettare un intervento;
- `unclassified_policy_error_cluster`: almeno tre casi `other`, migliorare prima la
  tassonomia;
- `sparse_policy_error_signal`: casi affidabili ma isolati, nessun training;
- `no_policy_error_signal`: nessun errore affidabile nelle fasi esposte.

Nessun tuning post-hoc di soglie, roster o seed e nessuna promozione v15 sono consentiti.

## Pilot pre-formale

Un pilot da 72 osservazioni e 16 determinizzazioni ha validato tempi, otto bucket di
fase/posizione e serializzazione. Non entra nell'evidenza finale: ha fatto emergere prima
del congelamento due correzioni conservative, cioe' il bilanciamento esplicito anche per
avversario e la separazione degli errori gia' coperti da PIMC/solver. Il run formale usa
un seed invariato (`20260720`) ma piu' campioni e intervalli al 99%.

## Esecuzione formale

```bash
uv run python scripts/probe_policy_regret.py \
  --model data/models/best_a2c_v14.npz \
  --belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
  --num-observations 192 \
  --max-games 400 \
  --determinizations 64 \
  --seed 20260720 \
  --min-regret-points 1.0 \
  --confidence-z 2.576 \
  --out benchmarks/experiments/policy_regret_v14_v0_20260714/policy_regret_v14_o192_d64_seed20260720.json
```

Il report atteso contiene tutte le decisioni, valori per carta, aggregati per fase,
posizione, avversario e layer runtime, top case, hash degli asset e verdetto automatico.

