# Ablation suit consistency v0 (2026-07-11)

> Screening causale, non promozione. Evidenza sintetica:
> [suit_consistency_v0.v1.json](../reports/evidence/suit_consistency_v0.v1.json).
> Artefatti completi locali: `benchmarks/experiments/suit_consistency_v0/`.

## Domanda e verdetto

Dopo il fallimento del paired A2C v0, abbiamo lasciato invariato il policy gradient
on-policy e aggiunto un obiettivo separato:

`beta * KL(stopgrad(policy(originale)) || policy(copia rinominata))`.

**Verdetto: GO a un run intermedio da 50k, non alla promozione.** La risposta è
dose-dipendente. Beta `0.1` riduce il flip medio da `18,32%` a **`15,64%`** e la JS da
`0,14124` a **`0,10402` bit**. Contro v13 resta neutro (`-0,08` punti/partita medi, CI di
ogni seed compatibile con zero). Il difetto è migliorato ma non risolto.

## Contratto implementato

- La traiettoria reale conserva integralmente actor, critic, advantage, anchor e reward.
- Una permutazione non identità viene campionata con RNG dedicato per ogni traiettoria.
- La distribuzione originale diventa un target stop-gradient e viene rimappata sulle 40
  azioni della copia.
- Solo actor head e trunk della copia ricevono il gradiente ausiliario; il critic non viene
  toccato.
- Il termine viene mediato sugli N step on-policy, senza dimezzare il gradiente A2C.
- `--suit-consistency-beta 0` non aggiunge metadata né consuma RNG: su 2.000 partite ha
  prodotto un `.npz` identico all'artefatto del trainer precedente.
- Il path `v4+belief` da 409 feature resta rifiutato finché le 40 feature belief embedded
  non avranno una trasformazione verificata.

Un test su batch congelato dimostra che un piccolo update di sola consistency riduce la
forward-KL verso il target e lascia a zero i gradienti del critic.

## Protocollo

Per ogni beta (`0.001`, `0.01`, `0.1`) sono stati allenati tre modelli da 10.000 partite,
seed `20260711..20260713`, partendo da v13. Ricetta, BC anchor v11, reward shaping, seat e
mix di avversari coincidono con i controlli paired v0. Ogni modello è passato nella sonda
canonica da 4.096 osservazioni x 24 permutazioni e in un direct match da 10.000 partite
seat-fair contro il controllo dello stesso seed. Beta `0.1` è stato inoltre confrontato
direttamente con v13.

## Simmetria

| beta | flip medio | JS media (bit) | stati con almeno un flip |
|---:|---:|---:|---:|
| controllo | 18,32% | 0,14124 | 51,20% |
| 0,001 | 18,00% | 0,13903 | 50,83% |
| 0,01 | 17,42% | 0,12940 | 49,23% |
| **0,1** | **15,64%** | **0,10402** | **44,94%** |

I tre seed di beta `0.1` misurano `15,68%`, `15,89%` e `15,36%`: il miglioramento non è
train-seed-specifico. Rispetto al controllo il calo è `-2,67` punti percentuali, circa
`-14,6%` relativo. È un effetto reale ma ancora lontano dal target operativo `<9%`.

## Forza policy-only

| confronto medio sui tre seed | differenza punti/partita |
|---|---:|
| beta 0,001 - controllo corrispondente | +0,02 |
| beta 0,01 - controllo corrispondente | +0,10 |
| beta 0,1 - controllo corrispondente | +0,27 |
| **beta 0,1 - v13** | **-0,08** |

Nel confronto diretto con v13, i tre valori sono `-0,14`, `+0,01`, `-0,11`; le CI95 per
seed sono `[-0,49; +0,21]`, `[-0,34; +0,35]`, `[-0,46; +0,25]`. Non c'è una regressione
misurabile, ma nemmeno una prova di maggiore forza. Il `+0,27` contro i controlli brevi va
trattato come segnale secondario, non sommato aritmeticamente ai match contro v13.

## Prossimo run

Il run 50k complessivo supera il limite operativo per una sessione agente. Questo comando
esegue i tre seed **in sequenza**, salva checkpoint e log separati e può essere lanciato dal
maintainer senza saturare la CPU con tre training contemporanei:

```bash
mkdir -p benchmarks/experiments/suit_consistency_v0_50k
nohup sh -c '
set -e
for SEED in 20260711 20260712 20260713; do
  uv run python scripts/train_a2c.py \
    --init data/models/best_a2c_v13.npz \
    --out "benchmarks/experiments/suit_consistency_v0_50k/consistency_beta0100_50k_seed${SEED}.npz" \
    --encoder-version v4 --rollout-engine fast --fast-rollout numba \
    --opponent-mix bc_model:0.15,bc_model_pimc_belief:0.40,bc_model_value_lookahead_8x8:0.20,heuristic_trump_saver:0.12,heuristic_v1:0.04,heuristic_v2:0.06,random:0.03 \
    --opponent-model data/models/best_a2c_v11.npz \
    --opponent-belief-model data/models/belief_v0_h128_50k_seed20260702.npz \
    --opponent-value-model data/models/value_v1_v4_fullgame_h128_seed20260718.npz \
    --opponent-value-max-unknown-cards 8 \
    --bc-anchor data/models/best_a2c_v11.npz --bc-anchor-beta 0.01 \
    --overkill-penalty-mode gap --overkill-penalty-beta 0.3 \
    --overkill-low-lead-points-max 2 --lr 0.0003 --entropy-beta 0.0005 \
    --value-coef 0.5 --gamma 1.0 --update-every 20 --seat-fair \
    --metrics-mode summary --num-games 50000 --seed "$SEED" \
    --suit-consistency-beta 0.1 --checkpoint-games 10000,30000,50000 \
    --checkpoint-dir benchmarks/experiments/suit_consistency_v0_50k \
    --checkpoint-prefix "consistency_beta0100_seed${SEED}" \
    > "benchmarks/experiments/suit_consistency_v0_50k/train_seed${SEED}.log" 2>&1
done
' > benchmarks/experiments/suit_consistency_v0_50k/driver.log 2>&1 &
```

Artefatti attesi per seed: checkpoint `10k`, `30k`, `50k`, modello finale e log. Nel log
controllare assenza di traceback/NaN, presenza di `Saved model` e andamento di `suit_kl`.
Il passo successivo è sondare tutti i checkpoint: serve capire la traiettoria, non soltanto
il valore finale. Solo un calo continuo verso `<9%` senza regressione policy-only autorizza
PIMC 16x8 e decision quality.
