# Diagnostica della salute A2C v0 (2026-07-14)

> Protocollo e soglie fissati prima del run. Esito: **in attesa del probe multi-seed**.

## Domanda

Prima di aggiungere normalizzazione, gradient clipping o un nuovo algoritmo, vogliamo
capire quale parte del training A2C corrente sia realmente instabile. Il test non cerca
un modello più forte: osserva tre training brevi e decide **una sola correzione** da
isolare nel passaggio successivo.

Il trainer condivide un livello nascosto tra due compiti:

- l'*actor* sceglie la carta;
- il *critic* prova a prevedere quanto sarà favorevole il resto della partita, così
  l'actor può distinguere una mossa migliore o peggiore delle attese.

Un fatto già verificato nel codice è importante: `train_a2c.py` azzera sempre `wv/bv`
all'avvio. Questo succede anche se il file passato con `--init` contiene già un critic.
La diagnostica registra il comportamento ma **non lo corregge**.

## Contratto passivo

Il flag `train_a2c.py --diagnostics-json <report.json>` registra per ogni optimizer
update soltanto aggregati numerici:

- distribuzione di return, valore previsto e advantage;
- errore quadratico ed explained variance del critic;
- tasso di attivazione delle unità ReLU, senza conservare osservazioni;
- norme dei gradienti per trunk, actor e critic;
- passo reale prodotto da Adam, assoluto e relativo alla norma dei parametri.

La sonda non usa carte nascoste, non salva mani o mazzo, non consuma RNG e non modifica
gradienti o pesi. Un test end-to-end pretende uguaglianza bit-per-bit tra un training con
sonda e lo stesso training senza sonda.

## Ricetta congelata

Tre seed indipendenti: `20260714`, `20260715`, `20260716`. Ogni run usa 2.000 partite,
batch da 20 partite e quindi 100 update. Le metriche del critic, degli advantage e delle
attivazioni sono lette sulla seconda metà dei 100 update; picchi e passi Adam sono
controllati sull'intero run.

La ricetta conserva la linea v13/v14:

- init e BC-anchor: `best_a2c_v14.npz`, beta anchor `0,01`;
- encoder v4, fast rollout Numba, seat alternata;
- opponent mix: modello puro `15%`, PIMC belief 16×8 `40%`, value-lookahead 8×8
  `20%`, conservatore di briscole `12%`, heuristic v1 `4%`, v2 `6%`, random `3%`;
- belief v0 ufficiale e value v1 full-game già usato nella ricetta di training;
- penalità overkill `gap`, beta `0,3`; learning rate, entropy, value coefficient e
  gamma restano ai default correnti.

Non sono presenti normalizzazione advantage, clipping, decay, riuso del critic o
schedule paired: aggiungerli ora renderebbe impossibile attribuire il risultato.

## Soglie di instradamento

Ogni seed deve passare ogni soglia; non si media via un run problematico.

| controllo | soglia fissata |
|---|---:|
| mediana explained variance critic, metà finale | `>= 0,10` |
| quota update finali con explained variance negativa | `<= 25%` |
| mediana `abs(media advantage) / std advantage` | `<= 0,25` |
| `p95 / mediana` della norma gradiente globale, intero run | `<= 5,0` |
| p95 passo relativo trunk, intero run | `<= 1%` |
| p95 passo relativo actor, intero run | `<= 1%` |
| massima quota unità mai attive in un update finale | `<= 75%` |
| mediana tasso medio di attivazione | tra `2%` e `98%` |

Sono soglie diagnostiche volutamente larghe, non gate di promozione. L'errore quadratico
del critic è riportato per dare scala al problema, ma non ha una soglia assoluta: dipende
dalla scala dei reward e va letto insieme all'explained variance.

## Decisione preregistrata

La prima condizione fallita decide il solo test successivo:

1. critic insufficiente → confronto `reset` contro `reuse` del critic;
2. critic sano ma advantage sbilanciato → normalizzazione sull'intero update;
3. segnali sani ma gradienti/passi con picchi → global gradient clipping;
4. attivazioni estreme → approfondire la rappresentazione prima dell'ottimizzatore;
5. tutti i controlli sani su tutti i seed → nessuna correzione numerica prioritaria;
   implementare e confrontare la schedule di training davvero paired.

Il probe non può promuovere i tre modelli temporanei né autorizzare un training lungo.

## Riproduzione

```bash
uv run python scripts/run_a2c_health_probe.py
```

Artefatti locali attesi in `benchmarks/experiments/a2c_health_v14_v0_20260714/`:

- tre modelli temporanei `.npz`;
- tre report per-update `.diagnostics.json`;
- `a2c_health_v14_g2000_3seeds.json`, con hash degli input, gate per seed e decisione.

Il comando accetta `--resume`: salta un seed soltanto se modello e diagnostica esistono
entrambi e il report conserva seed, dimensione, batch e hash init compatibili.
