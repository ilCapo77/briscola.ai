/**
 * Modulo API per Briscola AI
 * Gestisce tutta la comunicazione con il server backend
 */

const API = (() => {
    const STRINGS = {
        failedToCreateGame: 'Impossibile creare la partita',
        failedToGetGameState: 'Impossibile ottenere lo stato della partita',
        failedToPlayCard: 'Impossibile giocare la carta',
        failedToGetGameResult: 'Impossibile ottenere il risultato della partita',
        errorCreatingGame: 'Errore durante la creazione della partita',
        errorGettingGameState: 'Errore nel recupero dello stato della partita',
        errorPlayingCard: 'Errore durante la giocata della carta',
        errorGettingGameResult: 'Errore nel recupero del risultato della partita',
        wsEstablished: 'Connessione WebSocket stabilita',
        wsErrorParsing: 'Errore nel parsing del messaggio WebSocket',
        wsError: 'Errore WebSocket',
        wsClosed: 'Connessione WebSocket chiusa',
        wsReconnecting: 'Tentativo di riconnessione WebSocket...',
        intentionalDisconnect: 'Disconnessione intenzionale',
    };

    const _requireServedOverHttp = () => {
        if (window.location.protocol === 'file:') {
            throw new Error(
                'La UI deve essere servita dal server (non aprire index.html da filesystem). Avvia `briscola-server` e visita http://localhost:8000.'
            );
        }
    };

    // Usa sempre la stessa origin da cui è servita la UI.
    const API_URL = new URL('/api', window.location.origin).toString().replace(/\/$/, '');

    let gameId = null;
    let playerIndex = null;
    let websocket = null;
    let pingIntervalId = null;
    let reconnectTimeoutId = null;
    let reconnectAttempt = 0;
    let intentionalDisconnect = false;
    let currentOnMessage = null;
    let currentCallbacks = null;

    /**
     * Calcola un delay di riconnessione con backoff esponenziale (con jitter).
     *
     * Nota didattica:
     * - un retry "fisso" può martellare il server e creare burst di richieste
     * - un backoff riduce il carico e rende la UI più stabile in caso di rete ballerina
     */
    const _reconnectDelayMs = (attempt) => {
        const baseMs = 600;
        const maxMs = 10000;
        const factor = 1.6;
        const jitterPct = 0.2; // +/- 20%

        const exp = Math.min(maxMs, Math.round(baseMs * (factor ** Math.max(0, attempt - 1))));
        const jitter = exp * jitterPct * (Math.random() * 2 - 1);
        return Math.max(0, Math.round(exp + jitter));
    };

    const _closeActiveWebSocket = ({ resetGameInfo }) => {
        intentionalDisconnect = true;

        if (reconnectTimeoutId) {
            clearTimeout(reconnectTimeoutId);
            reconnectTimeoutId = null;
        }

        if (pingIntervalId) {
            clearInterval(pingIntervalId);
            pingIntervalId = null;
        }

        if (websocket && websocket.readyState !== WebSocket.CLOSED) {
            websocket.close(1000, STRINGS.intentionalDisconnect);
        }
        websocket = null;
        reconnectAttempt = 0;

        if (resetGameInfo) {
            API.gameId = null;
            API.playerIndex = null;
        }
    };

    /**
     * Crea una nuova partita sul server
     * @param {Object} config - Configurazione della partita
     * @returns {Promise} - Promise con l'esito della creazione
     */
    const createGame = async (config) => {
        _requireServedOverHttp();
        try {
            const response = await fetch(`${API_URL}/games`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToCreateGame);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorCreatingGame}:`, error);
            throw error;
        }
    };

    /**
     * Ottiene lo stato corrente di una partita
     * @param {string} gameId - ID della partita
     * @param {number} playerIndex - Indice del giocatore per una vista specifica
     * @returns {Promise} - Promise con lo stato della partita
     */
    const getGameState = async (gameId, playerIndex) => {
        _requireServedOverHttp();
        try {
            const url = new URL(`${API_URL}/games/${gameId}`);
            if (playerIndex !== undefined) {
                url.searchParams.append('player_index', playerIndex);
            }

            const response = await fetch(url);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToGetGameState);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorGettingGameState}:`, error);
            throw error;
        }
    };

    /**
     * Gioca una carta nella partita
     * @param {string} gameId - ID della partita
     * @param {number} playerIndex - Indice del giocatore
     * @param {number} cardIndex - Indice della carta nella mano del giocatore
     * @returns {Promise} - Promise con l'esito dell'azione
     */
    const playCard = async (gameId, playerIndex, cardIndex) => {
        _requireServedOverHttp();
        try {
            const response = await fetch(`${API_URL}/games/${gameId}/actions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    game_id: gameId,
                    player_index: playerIndex,
                    card_index: cardIndex
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToPlayCard);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorPlayingCard}:`, error);
            throw error;
        }
    };

    /**
     * Ottiene il risultato finale di una partita
     * @param {string} gameId - ID della partita
     * @returns {Promise} - Promise con il risultato della partita
     */
    const getGameResult = async (gameId) => {
        _requireServedOverHttp();
        try {
            const response = await fetch(`${API_URL}/games/${gameId}/result`);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || STRINGS.failedToGetGameResult);
            }

            return await response.json();
        } catch (error) {
            console.error(`${STRINGS.errorGettingGameResult}:`, error);
            throw error;
        }
    };

    /**
     * Connette al WebSocket per aggiornamenti in tempo reale
     * @param {string} gameId - ID della partita
     * @param {number} playerIndex - Indice del giocatore
     * @param {Function} onMessage - Callback per i messaggi ricevuti
     * @returns {WebSocket} - Connessione WebSocket
     */
    const connectWebSocket = (gameId, playerIndex, onMessage) => {
        _requireServedOverHttp();

        // Chiude un'eventuale connessione già esistente
        _closeActiveWebSocket({ resetGameInfo: false });

        // Determina l'URL del WebSocket (ws o wss in base a http/https), stessa origin della UI.
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsBase = `${wsProtocol}//${window.location.host}`;
        const wsUrl = new URL(`/api/ws/${gameId}/${playerIndex}`, wsBase).toString();

        intentionalDisconnect = false;
        currentOnMessage = onMessage;
        currentCallbacks = null;

        // Backward compatibility: se il terzo argomento è un oggetto, lo trattiamo come callbacks.
        if (onMessage && typeof onMessage === 'object') {
            currentCallbacks = onMessage;
            currentOnMessage = onMessage.onMessage;
        }

        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            console.log(STRINGS.wsEstablished);
            reconnectAttempt = 0;
            // Salva ID partita e indice giocatore per la riconnessione
            API.gameId = gameId;
            API.playerIndex = playerIndex;

            if (reconnectTimeoutId) {
                clearTimeout(reconnectTimeoutId);
                reconnectTimeoutId = null;
            }

            if (currentCallbacks && typeof currentCallbacks.onOpen === 'function') {
                currentCallbacks.onOpen();
            }

            // Invia un ping ogni 30 secondi per mantenere viva la connessione (evita leak: 1 solo interval).
            if (pingIntervalId) {
                clearInterval(pingIntervalId);
                pingIntervalId = null;
            }
            pingIntervalId = setInterval(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({ type: 'ping' }));
                }
            }, 30000);
        };

        websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                // Keepalive: il backend risponde ai ping con `{type: "pong"}`.
                // Questi messaggi NON sono uno snapshot dello stato di gioco: se li passiamo
                // al layer UI rischiamo di “resettare” la mano/punti a valori vuoti e bloccare
                // la partita fino al prossimo evento.
                if (data && typeof data === 'object' && (data.type === 'pong' || data.type === 'ping')) {
                    return;
                }

                if (typeof currentOnMessage === 'function') {
                    currentOnMessage(data);
                }
            } catch (error) {
                console.error(`${STRINGS.wsErrorParsing}:`, error);
            }
        };

        websocket.onerror = (error) => {
            console.error(`${STRINGS.wsError}:`, error);
            if (currentCallbacks && typeof currentCallbacks.onError === 'function') {
                currentCallbacks.onError(error);
            }
        };

        websocket.onclose = (event) => {
            console.log(`${STRINGS.wsClosed}:`, event.code, event.reason);

            if (pingIntervalId) {
                clearInterval(pingIntervalId);
                pingIntervalId = null;
            }

            if (currentCallbacks && typeof currentCallbacks.onClose === 'function') {
                currentCallbacks.onClose(event);
            }

            // Prova a riconnettersi dopo un ritardo se non è stata una chiusura intenzionale.
            if (!intentionalDisconnect) {
                reconnectAttempt += 1;
                const delayMs = _reconnectDelayMs(reconnectAttempt);

                if (currentCallbacks && typeof currentCallbacks.onReconnectAttempt === 'function') {
                    currentCallbacks.onReconnectAttempt({ attempt: reconnectAttempt, delayMs });
                }

                reconnectTimeoutId = setTimeout(() => {
                    if (API.gameId !== null && API.playerIndex !== null) {
                        console.log(STRINGS.wsReconnecting);
                        connectWebSocket(API.gameId, API.playerIndex, currentCallbacks || currentOnMessage);
                    }
                }, delayMs);
            }
        };

        return websocket;
    };

    /**
     * Disconnette dal WebSocket
     */
    const disconnectWebSocket = () => {
        _closeActiveWebSocket({ resetGameInfo: true });
    };

    // API pubblica
    return {
        gameId,
        playerIndex,
        createGame,
        getGameState,
        playCard,
        getGameResult,
        connectWebSocket,
        disconnectWebSocket
    };
})();
