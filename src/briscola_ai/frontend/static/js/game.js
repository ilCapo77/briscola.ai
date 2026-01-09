/**
 * Modulo di gioco per Briscola AI - Versione Semplificata
 *
 * Coordina la logica di gioco. La UI è ora guidata esclusivamente
 * dallo stato ricevuto via WebSocket dal backend.
 *
 * L'IA gioca automaticamente lato server (non più client-side).
 */

document.addEventListener('DOMContentLoaded', () => {
    // Game state - minimal, derived from backend
    const store = Store.create({
        gameId: null,
        playerName: null,
        playerIndex: 0,       // Human is always player 0
        opponentIndex: 1,     // AI is always player 1
        connected: false,
        observation: null,
        gameOver: false
    });

    const getState = () => store.getState();

    // Track last known table state to detect changes
    let lastTableCardsCount = 0;
    let lastTrickWinner = null;

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

        // Handle trick result message - shows both cards with winner
        if (data?.type === 'trick_result') {
            handleTrickResult(data);
            return;
        }

        // Handle AI card reveal - shows the AI's selected card face-up in its hand
        if (data?.type === 'ai_card_reveal') {
            console.log('AI card reveal:', data.card_index, data.card);
            UI.revealOpponentCard(data.card_index, data.card);
            return;
        }

        // Validate it's an observation
        if (!data || !Array.isArray(data.my_hand)) {
            console.warn('Ignoring invalid WS message:', data);
            return;
        }

        store.setState({
            observation: data,
            gameOver: !!data.game_over
        });

        updateUI(data);

        // Check game over
        if (data.game_over) {
            handleGameOver();
        }
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
        if (!state.observation?.my_turn || state.gameOver) return;

        try {
            await API.playCard(state.gameId, state.playerIndex, cardIndex);
            // UI update will come via WebSocket
        } catch (error) {
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
