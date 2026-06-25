# Struttura del package AI

Questo package contiene tutto cio' che riguarda gli agenti e la pipeline ML. E' diviso
per responsabilita', cosi' chi studia il progetto puo' partire dal livello giusto senza
dover leggere subito i path ottimizzati.

## Mappa

- `agents/`: agenti giocabili nel backend/UI. La facciata `__init__.py` esporta l'API pubblica;
  l'implementazione e' separata in `base.py`, `rule_based.py`, `hybrid_endgame.py`, `registry.py`.
- `models/`: caricamento modelli `.npz`, agente BC/A2C, catalogo server-side e provisioning.
- `endgame/`: solver esatto del finale 2-player a mazzo vuoto.
- `encoding/`: spazio azioni e encoder observation -> feature/mask per i modelli.
- `training/`: componenti di training condivisi (curriculum, reward shaping, opponent mix, regolarizzazioni).
- `evaluation/`: valutazione offline, matrici benchmark e metriche di qualita' decisionale.
- `fast/`: motore 2-player mutabile in Python/NumPy per rollout veloci.
- `numba/`: path JIT ad alto throughput. `core.py` contiene regole/euristiche numeriche,
  `observation.py` encoder e kernel condivisi, `mlp.py` wrapper MLP/A2C, `types.py` DTO.

## Regola didattica

Il dominio canonico resta in `briscola_ai.domain`. Gli agenti ricevono sempre
`PlayerObservation`, mai `GameState` completo, salvo moduli-oracolo espliciti come
`endgame.solver` che sono usati solo dopo ricostruzione lecita dell'informazione.

## Import

Il nuovo codice deve usare i percorsi organizzati sopra. I vecchi moduli root storici
sono stati rimossi per evitare ambiguita' didattica: ogni import deve rendere chiara
la responsabilita' del modulo che sta usando.
