# Struttura del package AI

Questo package contiene tutto cio' che riguarda gli agenti e la pipeline ML. E' diviso
per responsabilita', cosi' chi studia il progetto puo' partire dal livello giusto senza
dover leggere subito i path ottimizzati.

## Mappa

- `agents/`: agenti giocabili nel backend/UI e registry (`build_agent`, `list_agent_specs`).
- `models/`: caricamento modelli `.npz`, agente BC/A2C, catalogo server-side e provisioning.
- `endgame/`: solver esatto del finale 2-player a mazzo vuoto.
- `encoding/`: spazio azioni e encoder observation -> feature/mask per i modelli.
- `training/`: componenti di training condivisi (curriculum, reward shaping, opponent mix, regolarizzazioni).
- `evaluation/`: valutazione offline, matrici benchmark e metriche di qualita' decisionale.
- `fast/`: motore 2-player mutabile in Python/NumPy per rollout veloci.
- `numba/`: kernel Numba per rollout/evaluation ad alto throughput.

## Regola didattica

Il dominio canonico resta in `briscola_ai.domain`. Gli agenti ricevono sempre
`PlayerObservation`, mai `GameState` completo, salvo moduli-oracolo espliciti come
`endgame.solver` che sono usati solo dopo ricostruzione lecita dell'informazione.

## Compatibilita'

Alcuni moduli storici al livello root (`bc_model_agent.py`, `fast_2p.py`,
`evaluation_matrix.py`, ecc.) sono rimasti come shim. Servono per non rompere script,
test e import esterni durante la migrazione. Il nuovo codice dovrebbe preferire i
percorsi organizzati indicati sopra.
