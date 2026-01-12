/**
 * Modulo di gioco per Briscola AI - Versione Semplificata
 *
 * Coordina la logica di gioco. La UI è ora guidata esclusivamente
 * dallo stato ricevuto via WebSocket dal backend.
 *
 * Modello "standard":
 * - il backend avanza automaticamente la partita (incluse le mosse IA)
 * - il frontend controlla solo la presentazione (hold/animazioni) degli update ricevuti.
 */

document.addEventListener('DOMContentLoaded', () => {
    /**
     * Durata (ms) dell'evidenziazione "reveal" prima di applicare gli update UI.
     *
     * Obiettivo didattico/UX:
     * - rendere percepibile la sequenza degli eventi (carta scelta -> carta sul tavolo)
     * - mantenere la stessa durata per player e IA (coerenza visiva)
     *
     * Nota: la vera sorgente di verità resta il backend; qui "tratteniamo" solo il rendering.
     */
    const REVEAL_DURATION_MS = 1400;

    /**
     * Durata (ms) di visualizzazione del risultato presa (chi vince + punti).
     *
     * Nota architetturale:
     * il backend invia `trick_result` e subito dopo anche uno snapshot aggiornato.
     * Per evitare che il risultato “sparisca” immediatamente, il frontend trattiene
     * lo snapshot finché non è passato questo tempo.
     */
    const TRICK_RESULT_HOLD_MS = 2000;

    // Game state - minimal, derived from backend
    const store = Store.create({
        gameId: null,
        playerName: null,
        playerIndex: 0,       // Human is always player 0
        opponentIndex: 1,     // AI is always player 1
        connected: false,
        observation: null,
        gameOver: false,
        actionInFlight: false
    });

    const getState = () => store.getState();

    // Track last known table state to detect changes
    let lastTableCardsCount = 0;

    // UI hold: quando evidenziamo una carta, rinviamo il rendering dello snapshot
    // finché non è passato il tempo di reveal (evita che la carta appaia sul tavolo
    // mentre è ancora "in mano").
    let uiHoldUntilMs = 0;
    /**
     * Coda di eventi UI provenienti dal backend.
     *
     * Motivazione:
     * - Con il modello server-driven, gli eventi arrivano "subito" dal WS:
     *   snapshot, reveal IA, risultato presa, snapshot post-presa.
     * - Per mantenere una sequenza visiva didattica (carta 1 -> carta 2 -> risultato),
     *   il frontend mette in coda gli eventi e li consuma rispettando gli hold.
     *
     * Tipi attesi:
     * - { type: 'observation', data: <snapshot> }
     * - { type: 'ai_card_reveal', data: <message> }
     * - { type: 'trick_result', data: <message> }
     */
    let pendingEvents = [];
    let flushTimeoutId = null;

    const _scheduleFlush = () => {
        if (flushTimeoutId) {
            clearTimeout(flushTimeoutId);
            flushTimeoutId = null;
        }
        const delay = Math.max(0, uiHoldUntilMs - Date.now());
        flushTimeoutId = setTimeout(_flushPending, delay);
    };

    const _holdUiForReveal = () => {
        uiHoldUntilMs = Math.max(uiHoldUntilMs, Date.now() + REVEAL_DURATION_MS);
        _scheduleFlush();
    };

    const _holdUiForTrickResult = () => {
        uiHoldUntilMs = Math.max(uiHoldUntilMs, Date.now() + TRICK_RESULT_HOLD_MS);
        _scheduleFlush();
    };

    /**
     * Accoda un evento, collassando snapshot consecutivi.
     *
     * Gli snapshot (`observation`) sono ridondanti: se ne arrivano più di uno di fila
     * mentre siamo in hold, teniamo solo l'ultimo per evitare flicker e lavoro inutile.
     */
    const _enqueueEvent = (event) => {
        if (event.type === 'observation') {
            const last = pendingEvents[pendingEvents.length - 1];
            if (last?.type === 'observation') {
                pendingEvents[pendingEvents.length - 1] = event;
            } else {
                pendingEvents.push(event);
            }
        } else {
            pendingEvents.push(event);
        }
        _scheduleFlush();
    };

    const _applyObservation = (obs) => {
        store.setState({
            observation: obs,
            gameOver: !!obs.game_over,
            // La UI è guidata dallo stato server: quando applichiamo uno snapshot valido,
            // possiamo considerare "chiusa" l'azione locale (lock click).
            actionInFlight: false
        });

        updateUI(obs);

        if (obs.game_over) {
            handleGameOver();
        }
    };

    const _flushPending = () => {
        if (flushTimeoutId) {
            clearTimeout(flushTimeoutId);
            flushTimeoutId = null;
        }

        if (Date.now() < uiHoldUntilMs) {
            _scheduleFlush();
            return;
        }

        // Consuma quanti più eventi possibili finché non entriamo in un nuovo hold.
        while (pendingEvents.length > 0 && Date.now() >= uiHoldUntilMs) {
            const next = pendingEvents.shift();

            if (next.type === 'ai_card_reveal') {
                const data = next.data;
                console.log('AI card reveal:', data.card_index, data.card);
                UI.revealOpponentCard(data.card_index, data.card);
                _holdUiForReveal();
                break;
            }

            if (next.type === 'trick_result') {
                handleTrickResult(next.data);
                break;
            }

            if (next.type === 'observation') {
                _applyObservation(next.data);
                continue;
            }

            // Tipo sconosciuto: logghiamo per debug e proseguiamo.
            console.warn('Evento WS con tipo non gestito:', next.type, next);
        }

        if (pendingEvents.length > 0) _scheduleFlush();
    };

    /**
     * Update the entire UI from observation
     */
    const updateUI = (obs) => {
        const state = getState();

        // Player hand
        const isMyTurn = obs.my_turn && !obs.game_over;
        UI.renderPlayerHand(obs.my_hand || [], isMyTurn, playCard);

        // Opponent hand
        const opponentHandSize = obs[`player_${state.opponentIndex}_hand_size`] || 0;
        UI.renderOpponentHand(opponentHandSize);

        // Points
        UI.updatePlayerPoints(obs.my_points || 0);
        const opponentPoints = obs[`player_${state.opponentIndex}_points`] || 0;
        UI.updateOpponentInfo('Avversario IA', opponentPoints);

        // Table cards
        UI.renderTableCards(obs.table_cards || []);

        // Trump card
        UI.renderTrumpCard(obs.trump_card);

        // Deck count
        UI.updateDeckCount(obs.cards_remaining_in_deck || 0);

        // Turn message
        if (obs.game_over) {
            UI.showTurnMessage('Partita terminata');
        } else if (obs.my_turn) {
            UI.showTurnMessage('Tocca a te - scegli una carta');
        } else {
            UI.showTurnMessage('Avversario sta pensando...', true);
        }

        // Detect trick completion (table cleared)
        const currentTableCount = (obs.table_cards || []).length;
        if (lastTableCardsCount === 2 && currentTableCount === 0) {
            // A trick just completed - show result if we have winner info
            // The winner is determined by comparing points changes
            // For simplicity, just show that trick was completed
        }
        lastTableCardsCount = currentTableCount;
    };

    /**
     * Handle WebSocket messages
     */
    const handleGameUpdate = (data) => {
        // Ignore ping/pong
        if (data?.type === 'ping' || data?.type === 'pong') return;

        if (data?.type === 'ai_card_reveal') {
            _enqueueEvent({ type: 'ai_card_reveal', data });
            _flushPending();
            return;
        }

        if (data?.type === 'trick_result') {
            _enqueueEvent({ type: 'trick_result', data });
            _flushPending();
            return;
        }

        // Validate it's an observation
        if (!data || !Array.isArray(data.my_hand)) {
            console.warn('Ignoring invalid WS message:', data);
            return;
        }

        _enqueueEvent({ type: 'observation', data });
        _flushPending();
    };

    /**
     * Handle trick result - display both cards and winner
     */
    const handleTrickResult = (data) => {
        const state = getState();

        // Render both cards on the table
        UI.renderTableCards(data.trick_cards || []);

        // Remove any revealed card from hand (to avoid duplication: card on table AND in hand)
        UI.removeRevealedCard();

        // Show winner message
        const winnerLabel = data.winner_index === state.playerIndex ? 'Tu vinci!' : `${data.winner_name} vince!`;
        const pointsText = data.points > 0 ? ` (+${data.points} punti)` : '';
        UI.showTurnMessage(`${winnerLabel}${pointsText}`, false);

        // Trattieni la UI: lo snapshot “post presa” arriverà subito dopo, ma vogliamo
        // lasciare il tempo di leggere il risultato.
        _holdUiForTrickResult();
    };

    /**
     * Start a new game
     */
    const startGame = async (config) => {
        try {
            const playerNames = [config.playerName, 'Avversario IA'];

            const result = await API.createGame({
                num_players: 2,
                player_names: playerNames
            });

            store.setState({
                gameId: result.game_id,
                playerName: config.playerName,
                playerIndex: 0,
                opponentIndex: 1,
                connected: false,
                observation: null,
                gameOver: false
            });

            UI.setPlayerName(config.playerName);
            UI.updateGameInfo({ gameId: result.game_id, connected: false });
            UI.showGameBoard();

            // Connect WebSocket
            API.connectWebSocket(result.game_id, 0, {
                onMessage: handleGameUpdate,
                onOpen: () => {
                    store.setState({ connected: true });
                    UI.updateGameInfo({ connected: true });
                },
                onClose: () => {
                    store.setState({ connected: false });
                    UI.updateGameInfo({ connected: false });
                }
            });

        } catch (error) {
            alert(`Errore: ${error.message}`);
        }
    };

    /**
     * Play a card
     */
    const playCard = async (cardIndex) => {
        const state = getState();
        if (!state.observation?.my_turn || state.gameOver || state.actionInFlight) return;

        try {
            // Feedback immediato: evidenziamo la carta scelta prima che venga "spostata"
            // sul tavolo tramite update WebSocket (effetto simile al reveal dell'IA).
            store.setState({ actionInFlight: true });
            UI.revealPlayerCard(cardIndex);
            _holdUiForReveal();

            await API.playCard(state.gameId, state.playerIndex, cardIndex);
            // UI update will come via WebSocket
        } catch (error) {
            // In caso di errore, sblocchiamo la UI: lo snapshot potrebbe non arrivare.
            store.setState({ actionInFlight: false });
            // Ripristina la mano "normale" (rimuove highlight/disabled) ri-renderizzando dallo stato corrente.
            if (state.observation) updateUI(state.observation);
            alert(`Errore: ${error.message}`);
        }
    };

    /**
     * Handle game over
     */
    const handleGameOver = async () => {
        const state = getState();

        try {
            const result = await API.getGameResult(state.gameId);
            UI.displayGameResult(result);
        } catch (error) {
            console.error('Failed to get result:', error);
            UI.displayGameResult({
                winner: 'Errore',
                points: {}
            });
        }

        API.disconnectWebSocket();
        store.setState({ connected: false });
        UI.updateGameInfo({ connected: false });
    };

    /**
     * Reset and start over
     */
    const resetGame = () => {
        API.disconnectWebSocket();

        store.setState({
            gameId: null,
            playerName: null,
            playerIndex: 0,
            opponentIndex: 1,
            connected: false,
            observation: null,
            gameOver: false
        });

        lastTableCardsCount = 0;
        UI.showGameSetup();
    };

    // Initialize
    UI.init({
        onStartGame: startGame,
        onNewGame: resetGame
    });

    UI.showGameSetup();
});
