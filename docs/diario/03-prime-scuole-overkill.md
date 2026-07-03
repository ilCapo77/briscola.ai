# Approfondimento — BC, RL e la guerra all'overkill

**Capitolo del diario:** [Capitolo 3](https://ai.briscola.dev/diario) · **Periodo:** gennaio–febbraio 2026

## Le due scuole

- **Behavioral Cloning** (`45cc792`, `aeeafc1`): encoder osservazione → MLP → azione tra 40
  id carta con action mask. Impara in fretta, copia anche i difetti del teacher.
- **Reinforcement Learning**: policy gradient (`cb5a0b1`) poi A2C con reward shaping
  (`4c12e37`) su delta punti per presa. Primo `best_a2c`: **+9.71** su heuristic_v1
  (big holdout), cresciuto a ~+12.69 con league training (avversario campione congelato
  nel mix per evitare il "chasing" di due policy che cambiano insieme, `3da6a17`).

## La guerra all'overkill (10 febbraio)

Overkill = giocare una briscola alta per vincere una presa povera. Il best dell'epoca lo
faceva nel **20.3%** delle occasioni. Tre tentativi:

| Rimedio | Commit | Esito |
|---|---|---|
| Penalità flat in training | `ae94aa9` | "NON riduce in modo affidabile" |
| Penalità proporzionale al gap | `92f4f5f` | "NON migliora (tende anzi a peggiorarla)" |
| **Guard in inference** (post-processing: scegli la briscola minima che vince comunque) | `6cd1d23`, A/B `bdefa6a` | **Adottato: costo in forza ≈ 0** |

Lezione: correggere un vizio nella funzione di reward può distorcere tutto il resto;
un vincolo esplicito al momento della decisione è chirurgico.
