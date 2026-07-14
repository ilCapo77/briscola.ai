# Training A2C realmente paired v0 (2026-07-14)

> Protocollo e soglie fissati prima del run multi-seed. Esito: **in attesa del job
> 20k/40k**.

## Domanda

Il vecchio `--seat-fair` alterna il posto della policy, ma ogni partita riceve un mazzo
nuovo e ricampiona l'avversario. I due esempi consecutivi non controllano quindi la
stessa situazione da entrambi i lati.

Vogliamo capire se un training realmente appaiato riduca la dipendenza dalla fortuna del
mazzo e la variabilità tra run, senza indebolire il modello. Non assumiamo che sia meglio:
vedere due volte lo stesso ambiente riduce infatti la varietà dei mazzi a parità di
partite.

## Contratto implementato

`train_a2c.py --training-schedule paired` costruisce l'intera schedule prima del primo
update. Ogni coppia adiacente:

- usa lo stesso `game_seed`, quindi lo stesso mazzo e la stessa distribuzione iniziale;
- usa lo stesso tipo di avversario campionato dall'opponent mix;
- assegna la policy prima alla seat 0 e poi alla seat 1;
- resta interamente nello stesso optimizer update, quindi vede gli stessi pesi.

Le due partite non devono avere le stesse mosse: scambiando la seat cambiano mano,
osservazioni e sviluppo della partita. Il pairing controlla ambiente e avversario, non
forza traiettorie artificialmente uguali.

La modalità `serial` resta il default e conserva l'alternanza storica di `--seat-fair`.
La modalità paired rifiuta:

- `num_games` dispari;
- `update_every` dispari;
- un numero di partite non multiplo di `update_every`, perché il trainer storico non
  applica un update finale parziale.

Il modello conserva nei metadati modalità, numero di ambienti, ordine seat, seed degli
RNG e SHA-256 della schedule. Dominio, fast Python e batch Numba leggono lo stesso oggetto
schedule. I test coprono i tre path, la riproducibilità, le coppie non spezzate e
l'uguaglianza bit-per-bit tra default seriale e `--training-schedule serial` esplicito.

## Tre regimi

Per ognuno dei seed `20260717`, `20260718`, `20260719`:

| regime | partite | estrazioni ambiente (mazzo + opponent) | update |
|---|---:|---:|---:|
| `serial_same_games` | 20.000 | 20.000 | 1.000 |
| `paired_same_games` | 20.000 | 10.000 | 1.000 |
| `paired_same_decks` | 40.000 | 20.000 | 2.000 |

Il runner ricostruisce indipendentemente le schedule e rifiuta il job se:

- gli ambienti di `paired_same_games` non coincidono col prefisso dei primi 10.000
  ambienti seriali;
- i 20.000 ambienti di `paired_same_decks` non coincidono esattamente, nello stesso
  ordine, con quelli seriali.

La ricetta A2C resta quella del probe di salute: init e anchor v14, encoder v4, fast
Numba, PIMC belief 16×8 al 40%, modello puro 15%, value-lookahead 20%, conservatore 12%,
heuristic v1/v2 e random per il resto, beta anchor `0,01` e penalità overkill gap `0,3`.
Critic reset, advantage, clipping e learning rate non cambiano.

## Valutazione

Ogni modello temporaneo viene valutato su 4.000 partite seat-fair deterministiche,
tutte sulla suite `range(4_000_000, 4_002_000)`:

- contro v14, per misurare forza finale e dispersione tra i tre seed;
- direttamente contro il controllo seriale dello stesso seed.

Il report registra inoltre coefficiente di variazione della norma gradiente, dispersione
della media degli advantage, explained variance del critic, distanza dall'init e tempo.
Le CI95 delle valutazioni usano la coppia di partite come unità statistica.

## Gate preregistrato

Il confronto primario è `paired_same_games` contro `serial_same_games`. È **GO a un solo
screen paired più lungo** soltanto se valgono insieme:

1. mediana dei tre direct match paired-minus-serial `>= 0` punti;
2. almeno due seed su tre non negativi;
3. deviazione standard tra seed della forza contro v14 non superiore al seriale;
4. mediana del coefficiente di variazione dei gradienti non superiore al seriale.

È **STOP paired** se:

- la mediana diretta è `<= -0,25` punti e almeno due seed sono negativi; oppure
- sia la dispersione di forza sia quella dei gradienti crescono di almeno il `10%`.

Negli altri casi il verdetto è **inconcludente e serial resta default**. Il regime a pari
mazzi è solo di supporto: usa il doppio delle partite e non può salvare un fallimento del
confronto primario.

Nessuno dei nove modelli può essere promosso: il probe misura la schedule, non produce
v15.

## Esecuzione

Il job previsto supera cinque minuti. Preparazione e avvio:

```bash
mkdir -p benchmarks/experiments/a2c_paired_schedule_v0_20260714
nohup uv run python scripts/run_a2c_paired_schedule_probe.py --resume \
  > benchmarks/experiments/a2c_paired_schedule_v0_20260714/run.log 2>&1 &
```

Controllo del log:

```bash
tail -f benchmarks/experiments/a2c_paired_schedule_v0_20260714/run.log
```

Artefatto finale atteso:

`benchmarks/experiments/a2c_paired_schedule_v0_20260714/a2c_paired_schedule_g20000_3seeds.json`.

`--resume` verifica hash di modelli, diagnostiche e input delle valutazioni prima di
saltare un sottopasso già completato.
