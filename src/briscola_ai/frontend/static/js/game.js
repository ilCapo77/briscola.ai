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
    const AI_PLAYER_DISPLAY_NAME = 'Giocatore AI';

    /**
     * Durata (ms) di visualizzazione del risultato della mano (chi vince + punti).
     *
     * Nota architetturale:
     * il backend invia `trick_result` e subito dopo anche uno snapshot aggiornato.
     * Per evitare che il risultato “sparisca” immediatamente, il frontend trattiene
     * lo snapshot finché non è passato questo tempo.
     */
    const TRICK_RESULT_HOLD_MS = 2000;

    /**
     * Identificatore pseudonimo del client (persistente in localStorage).
     *
     * Obiettivo:
     * - poter fare split train/val "per giocatore" senza salvare nomi o PII nel DB.
     * - avere un id stabile tra partite, ma non riconducibile a una persona.
     */
    const _getClientId = () => {
        try {
            const key = 'briscola_client_id';
            let value = window.localStorage.getItem(key);
            if (value && typeof value === 'string') return value;
            value = (window.crypto && typeof window.crypto.randomUUID === 'function')
                ? window.crypto.randomUUID()
                : `client_${Math.random().toString(16).slice(2)}_${Date.now()}`;
            window.localStorage.setItem(key, value);
            return value;
        } catch (e) {
            // Fallback: se localStorage non è disponibile, usiamo un id effimero.
            return `client_${Math.random().toString(16).slice(2)}_${Date.now()}`;
        }
    };

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

    // Metadati runtime del server (es. modalità raccolta dati).
    let serverMeta = { dataset_requires_consent: false };

    const loadAiAgentMetadata = async () => {
        try {
            const meta = await API.getServerMeta();
            serverMeta = meta && typeof meta === 'object' ? meta : serverMeta;
            const required = meta?.dataset_requires_consent === true;
            UI.setDataCollectionConsent({
                required,
                description_it: required
                    ? 'Questa istanza sta raccogliendo dataset umano: le mosse verranno registrate in modo anonimo.'
                    : ''
            });
        } catch (error) {
            // Se non riusciamo a caricare i meta, manteniamo la UI in modalità "no consent required".
            serverMeta = { dataset_requires_consent: false };
            UI.setDataCollectionConsent({ required: false, description_it: '' });
        }

        try {
            const agents = await API.getAiAgents();
            UI.setAiAgents(agents);
        } catch (error) {
            console.warn('Impossibile caricare metadati agenti IA:', error);
        }

        try {
            const models = await API.getAiModels();
            UI.setAiModels(models);
        } catch (error) {
            console.warn('Impossibile caricare lista modelli IA (.npz):', error);
            UI.setAiModels({ models: [] });
        }
    };

    // Track last known table state to detect changes
    let lastTableCardsCount = 0;
    let lastAppliedServerVersion = -1;
    let pollingIntervalId = null;
    let pollingInFlight = false;

    // Timing umano (client-side): stimiamo il tempo decisionale (ms) come il tempo trascorso
    // da quando la UI applica uno snapshot in cui `my_turn=true` fino al click.
    //
    // Nota:
    // - usiamo "apply time" (non "receive time") perché il frontend può trattenere snapshot per UX (hold).
    // - misuriamo solo quando parte un turno umano (transizione my_turn false -> true).
    let my_turn_started_at_ms = null;
    let my_turn_observation_server_version = null;
    let was_my_turn = false;

    // UI hold: quando evidenziamo una carta, rinviamo il rendering dello snapshot
    // finché non è passato il tempo di reveal (evita che la carta appaia sul tavolo
    // mentre è ancora "in mano").
    let uiHoldUntilMs = 0;
    /**
     * Coda di eventi UI provenienti dal backend.
     *
     * Motivazione:
     * - Con il modello server-driven, gli eventi arrivano "subito" dal WS:
     *   snapshot, reveal IA, risultato mano, snapshot post-mano.
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

    const _displayNameForPlayer = (playerIndex, fallbackName = null) => {
        if (playerIndex === getState().playerIndex) return 'Tu';
        if (playerIndex === getState().opponentIndex) return AI_PLAYER_DISPLAY_NAME;
        return fallbackName || `Giocatore ${playerIndex + 1}`;
    };

    const _normalizeResultDisplayNames = (result) => {
        if (!result || typeof result !== 'object' || result.is_team_game) return result;

        const state = getState();
        const obsPlayers = state.observation?.players || [];
        const humanName = obsPlayers.find(p => p.index === state.playerIndex)?.name || state.playerName || 'Tu';
        const aiBackendName = obsPlayers.find(p => p.index === state.opponentIndex)?.name || null;
        const normalized = { ...result };

        if (result.winner_index === state.opponentIndex) {
            normalized.winner = AI_PLAYER_DISPLAY_NAME;
        } else if (result.winner_index === state.playerIndex) {
            normalized.winner = humanName;
        }

        if (result.points && typeof result.points === 'object') {
            normalized.points = {};
            for (const [name, points] of Object.entries(result.points)) {
                const label = name === aiBackendName ? AI_PLAYER_DISPLAY_NAME : name;
                normalized.points[label] = points;
            }
        }
        return normalized;
    };

    /**
     * Modalità debug: fallback polling al posto del WebSocket.
     *
     * Attivazione:
     * - aggiungi `?polling=1` all'URL della UI (es. http://localhost:8000/?polling=1)
     *
     * Motivazione:
     * - utile quando stai debuggando problemi di rete/reconnect e vuoi un flusso più "semplice"
     * - non è pensato come modalità principale (il WS resta il path normale)
     */
    const _pollingEnabledByUrl = () => {
        const params = new URLSearchParams(window.location.search);
        const value = params.get('polling');
        return value === '1' || value === 'true';
    };

    const _stopPolling = () => {
        if (pollingIntervalId) {
            clearInterval(pollingIntervalId);
            pollingIntervalId = null;
        }
        pollingInFlight = false;
    };

    const _startPolling = (gameId, playerIndex) => {
        _stopPolling();
        UI.updateGameInfo({ connected: false, statusText: 'Polling (debug)', statusClass: 'connecting' });

        const pollOnce = async () => {
            if (pollingInFlight) return;
            pollingInFlight = true;
            try {
                const obs = await API.getGameState(gameId, playerIndex);
                handleGameUpdate(obs);
            } catch (error) {
                console.warn('Polling error:', error);
                UI.updateGameInfo({ connected: false, statusText: 'Polling: errore rete', statusClass: 'reconnecting' });
            } finally {
                pollingInFlight = false;
            }
        };

        // Primo fetch immediato, poi intervallo.
        pollOnce();
        pollingIntervalId = setInterval(pollOnce, 700);
    };

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
        // Guard rail: se arrivano snapshot fuori ordine (reconnect/ritardi), ignoriamo quelli vecchi.
        const serverVersion = typeof obs?.server_version === 'number' ? obs.server_version : -1;
        if (serverVersion !== -1 && serverVersion <= lastAppliedServerVersion) {
            console.warn('Ignoring stale observation:', { serverVersion, lastAppliedServerVersion, obs });
            return;
        }
        if (serverVersion !== -1) lastAppliedServerVersion = serverVersion;

        store.setState({
            observation: obs,
            gameOver: !!obs.game_over,
            // La UI è guidata dallo stato server: quando applichiamo uno snapshot valido,
            // possiamo considerare "chiusa" l'azione locale (lock click).
            actionInFlight: false
        });

        // Tracking decision time: segna inizio turno umano quando `my_turn` diventa true.
        // Non resettiamo se arrivano snapshot successivi nello stesso turno (evita di sottostimare).
        if (obs && typeof obs.my_turn === 'boolean') {
            if (obs.my_turn && !was_my_turn) {
                my_turn_started_at_ms = Date.now();
                my_turn_observation_server_version = typeof obs.server_version === 'number' ? obs.server_version : null;
            }
            was_my_turn = obs.my_turn;
        }

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

        // Opponent hand and points (nuovo formato: array `players`)
        const opponent = (obs.players || []).find(p => p.index === state.opponentIndex);
        const opponentHandSize = opponent?.hand_size || 0;
        const opponentPoints = opponent?.points || 0;
        const opponentName = _displayNameForPlayer(state.opponentIndex, opponent?.name || 'Avversario IA');
        UI.renderOpponentHand(opponentHandSize);

        // Points
        UI.updatePlayerPoints(obs.my_points || 0);
        UI.updateOpponentInfo(opponentName, opponentPoints);

        // Table cards
        UI.renderTableCards(obs.table_cards || []);

        // Briscola: quando `trump_card` è null (es. deck vuoto) mostriamo comunque il seme.
        UI.renderTrumpCard(obs.trump_card, obs.trump_suit);

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

        // Contratto WS: gli snapshot devono avere `type: "observation"`.
        if (data?.type !== 'observation') {
            console.warn('Unhandled WS message type:', data?.type, data);
            return;
        }

        // Validate it's an observation
        if (!Array.isArray(data.my_hand)) {
            console.warn('Ignoring invalid observation (no my_hand):', data);
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
        const winnerName = _displayNameForPlayer(data.winner_index, data.winner_name);
        const winnerLabel = data.winner_index === state.playerIndex ? 'Tu vinci!' : `${winnerName} vince!`;
        const pointsText = data.points > 0 ? ` (+${data.points} punti)` : '';
        UI.showTurnMessage(`${winnerLabel}${pointsText}`, false);

        // Trattieni la UI: lo snapshot “post mano” arriverà subito dopo, ma vogliamo
        // lasciare il tempo di leggere il risultato.
        _holdUiForTrickResult();
    };

    /**
     * Start a new game
     */
    const startGame = async (config) => {
        try {
            const aiAgent = config.aiAgent || 'random';
            const aiModelId = config.aiModelId || null;
            const aiModelCompatible = config.aiModelCompatible === true;
            const aiModelCompatibilityReasonIt = config.aiModelCompatibilityReasonIt || null;

            // Il nome del player entra negli snapshot e nei messaggi di partita
            // (es. "X vince!"): teniamolo corto anche quando il modello selezionato
            // ha una label descrittiva molto lunga. La scelta dell'agente/modello resta
            // tracciata da `ai_agent` e `ai_model_id` nel payload.
            const playerNames = [config.playerName, AI_PLAYER_DISPLAY_NAME];

            if (serverMeta?.dataset_requires_consent === true && config.consentToDataCollection !== true) {
                throw new Error('Devi accettare la raccolta dati (anonima) per avviare la partita.');
            }

            if (aiAgent === 'bc_model' && !aiModelId) {
                throw new Error('Seleziona un modello (.npz) prima di avviare la partita.');
            }
            if (aiAgent === 'bc_model' && !aiModelCompatible) {
                const reason = aiModelCompatibilityReasonIt ? `\n\nMotivo: ${aiModelCompatibilityReasonIt}` : '';
                throw new Error(`Il modello selezionato non è compatibile.${reason}`);
            }

            const createPayload = {
                num_players: 2,
                player_names: playerNames,
                ai_agent: aiAgent,
                client_id: _getClientId(),
                consent_to_data_collection: config.consentToDataCollection === true,
            };
            if (aiAgent === 'bc_model') {
                createPayload.ai_model_id = aiModelId;
            }

            const result = await API.createGame(createPayload);

            store.setState({
                gameId: result.game_id,
                playerName: config.playerName,
                playerIndex: 0,
                opponentIndex: 1,
                connected: false,
                observation: null,
                gameOver: false
            });

            my_turn_started_at_ms = null;
            my_turn_observation_server_version = null;
            was_my_turn = false;

            UI.setPlayerName(config.playerName);
            UI.updateGameInfo({ gameId: result.game_id, connected: false, statusText: 'Connessione...', statusClass: 'connecting' });
            UI.showGameBoard();

            if (_pollingEnabledByUrl()) {
                // Modalità debug: niente WS, solo polling.
                _startPolling(result.game_id, 0);
            } else {
                // Connect WebSocket (path normale)
                API.connectWebSocket(result.game_id, 0, {
                    onMessage: handleGameUpdate,
                    onOpen: () => {
                        _stopPolling();
                        store.setState({ connected: true });
                        UI.updateGameInfo({ connected: true, statusText: 'Connesso', statusClass: 'connected' });
                    },
                    onClose: () => {
                        // Se la connessione cade durante un'azione, sblocchiamo la UI e ripristiniamo la mano.
                        const current = getState();
                        store.setState({ connected: false, actionInFlight: false });
                        UI.resetPlayerHandHighlights();
                        if (current.observation) updateUI(current.observation);

                        // Reset del buffer eventi: dopo reconnect useremo solo lo snapshot fresh dal server.
                        pendingEvents = [];
                        uiHoldUntilMs = 0;

                        UI.updateGameInfo({ connected: false, statusText: 'Non connesso', statusClass: 'disconnected' });
                    },
                    onReconnectAttempt: ({ attempt, delayMs }) => {
                        UI.updateGameInfo({
                            connected: false,
                            statusText: `Riconnessione... (tentativo ${attempt})`,
                            statusClass: 'reconnecting'
                        });
                        console.log(`WS reconnect attempt ${attempt} in ${delayMs}ms`);
                    }
                });
            }

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

            const decisionTimeMs = my_turn_started_at_ms != null ? (Date.now() - my_turn_started_at_ms) : null;
            await API.playCard(state.gameId, state.playerIndex, cardIndex, {
                observedServerVersion: my_turn_observation_server_version,
                decisionTimeMs
            });
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
            UI.displayGameResult(_normalizeResultDisplayNames(result));
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
        _stopPolling();

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
        lastAppliedServerVersion = -1;
        pendingEvents = [];
        uiHoldUntilMs = 0;
        my_turn_started_at_ms = null;
        my_turn_observation_server_version = null;
        was_my_turn = false;
        UI.showGameSetup();
    };

    // Initialize
    UI.init({
        onStartGame: startGame,
        onNewGame: resetGame
    });

    loadAiAgentMetadata();

    UI.showGameSetup();
});
