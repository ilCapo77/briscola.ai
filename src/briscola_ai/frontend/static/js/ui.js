/**
 * Modulo UI per Briscola AI - Versione Semplificata
 *
 * Gestisce il rendering della UI. Non contiene logica di gioco.
 */

const UI = (() => {
    const CARD_ASSET_BASE = '/static/assets/cards';

    // Map rank names to numbers for image paths
    const RANK_TO_NUMBER = {
        ACE: 1, TWO: 2, THREE: 3, FOUR: 4, FIVE: 5,
        SIX: 6, SEVEN: 7, JACK: 8, KNIGHT: 9, KING: 10
    };

    // DOM elements cache
    const elements = {
        gameSetup: document.getElementById('game-setup'),
        gameBoard: document.getElementById('game-board'),
        gameResult: document.getElementById('game-result'),
        gameForm: document.getElementById('game-form'),
        playerNameInput: document.getElementById('player-name-input'),
        gameId: document.getElementById('game-id'),
        gameStatus: document.getElementById('game-status'),
        opponentName: document.getElementById('opponent-name'),
        opponentPoints: document.getElementById('opponent-points'),
        opponentHand: document.getElementById('opponent-hand'),
        playerNameDisplay: document.getElementById('player-name-display'),
        playerPoints: document.getElementById('player-points'),
        playerHand: document.getElementById('player-hand'),
        turnIndicator: document.getElementById('turn-indicator'),
        deck: document.getElementById('deck'),
        deckCount: document.getElementById('deck-count'),
        trumpCard: document.getElementById('trump-card'),
        tableCards: document.getElementById('table-cards'),
        turnMessage: document.getElementById('turn-message'),
        trickResult: document.getElementById('trick-result'),
        resultContent: document.getElementById('result-content'),
        newGame: document.getElementById('new-game')
    };

    /**
     * Normalize card data from various backend formats
     */
    const _normalizeCard = (card) => {
        if (!card || typeof card !== 'object') return null;

        const suit = card.suit?.value || card.suit;
        const rankName = card.rank?.name || card.rank;
        const number = card.number || RANK_TO_NUMBER[rankName] || null;

        return { suit, number, points: card.points };
    };

    /**
     * Get card image source
     */
    const _cardImageSrc = (card) => {
        const normalized = _normalizeCard(card);
        if (!normalized?.suit || !normalized?.number) return null;
        return `${CARD_ASSET_BASE}/${normalized.suit}_${normalized.number}.png`;
    };

    /**
     * Create a card element
     */
    const createCardElement = (card, onClick = null) => {
        const cardEl = document.createElement('div');
        cardEl.className = 'card';

        if (!card) {
            // Face down card
            cardEl.classList.add('card-back');
            return cardEl;
        }

        const src = _cardImageSrc(card);
        if (src) {
            const img = document.createElement('img');
            img.className = 'card-face';
            img.src = src;
            img.alt = `Carta`;
            img.loading = 'lazy';
            cardEl.appendChild(img);
        } else {
            cardEl.classList.add('card-back');
        }

        if (onClick) {
            cardEl.classList.add('clickable');
            cardEl.addEventListener('click', onClick);
        } else {
            cardEl.classList.add('disabled');
        }

        return cardEl;
    };

    // --- Public API ---

    const init = (callbacks) => {
        elements.gameForm.addEventListener('submit', (e) => {
            e.preventDefault();
            callbacks.onStartGame?.({
                playerName: elements.playerNameInput.value || 'Giocatore'
            });
        });

        elements.newGame.addEventListener('click', () => {
            callbacks.onNewGame?.();
        });
    };

    const showGameSetup = () => {
        elements.gameSetup.classList.remove('hidden');
        elements.gameBoard.classList.add('hidden');
        elements.gameResult.classList.add('hidden');
    };

    const showGameBoard = () => {
        elements.gameSetup.classList.add('hidden');
        elements.gameBoard.classList.remove('hidden');
        elements.gameResult.classList.add('hidden');
    };

    const showGameResult = () => {
        elements.gameSetup.classList.add('hidden');
        elements.gameBoard.classList.add('hidden');
        elements.gameResult.classList.remove('hidden');
    };

    const updateGameInfo = ({ gameId, connected }) => {
        if (gameId) {
            elements.gameId.textContent = `ID: ${gameId.substring(0, 8)}...`;
        }
        if (connected !== undefined) {
            elements.gameStatus.textContent = connected ? 'Connesso' : 'Non connesso';
            elements.gameStatus.className = connected ? 'connected' : '';
        }
    };

    const renderPlayerHand = (cards, isMyTurn, onCardClick) => {
        elements.playerHand.innerHTML = '';
        cards.forEach((card, index) => {
            const onClick = isMyTurn ? () => onCardClick(index) : null;
            const cardEl = createCardElement(card, onClick);
            cardEl.classList.add('card-appear');
            elements.playerHand.appendChild(cardEl);
        });

        // Show/hide turn indicator
        elements.turnIndicator.style.display = isMyTurn ? 'inline-block' : 'none';
    };

    const renderOpponentHand = (cardCount) => {
        elements.opponentHand.innerHTML = '';
        for (let i = 0; i < cardCount; i++) {
            const cardEl = createCardElement(null);
            elements.opponentHand.appendChild(cardEl);
        }
    };

    /**
     * Reveal a specific card in opponent's hand (show face-up with highlight)
     */
    const revealOpponentCard = (cardIndex, card) => {
        const cards = elements.opponentHand.children;
        console.log('revealOpponentCard called:', cardIndex, 'cards in hand:', cards.length);
        if (cardIndex >= 0 && cardIndex < cards.length) {
            const cardEl = cards[cardIndex];
            // Replace card back with face-up card
            const src = _cardImageSrc(card);
            console.log('Revealing card with src:', src);
            if (src) {
                cardEl.classList.remove('card-back');
                cardEl.classList.add('revealed');
                const img = document.createElement('img');
                img.className = 'card-face';
                img.src = src;
                img.alt = 'Carta IA';
                cardEl.innerHTML = '';
                cardEl.appendChild(img);
            }
        } else {
            console.warn('Card index out of range:', cardIndex, 'vs', cards.length);
        }
    };

    /**
     * Evidenzia (lampeggia) la carta scelta dal giocatore nella sua mano.
     *
     * Nota didattica:
     * - il backend è la "single source of truth": questa è solo una micro-animazione
     *   locale per rendere chiaro quale carta è stata selezionata PRIMA che venga
     *   renderizzata sul tavolo tramite l'update WebSocket.
     * - non rimuove la carta dalla mano: la rimozione/aggiornamento arriva dallo snapshot.
     *
     * @param {number} cardIndex - indice della carta nella mano del player
     */
    const revealPlayerCard = (cardIndex) => {
        const cards = elements.playerHand.children;
        if (cardIndex < 0 || cardIndex >= cards.length) return;

        // Metti in evidenza la carta scelta e disabilita visivamente le altre
        // durante l'azione (evita confusione/doppi click).
        Array.from(cards).forEach((cardEl, idx) => {
            cardEl.classList.toggle('revealed', idx === cardIndex);
            cardEl.classList.toggle('disabled', idx !== cardIndex);
        });
    };

    /**
     * Remove any revealed card from the opponent's hand.
     * Use this when the card moves to the table.
     */
    const removeRevealedCard = () => {
        const revealedCards = elements.opponentHand.querySelectorAll('.revealed');
        revealedCards.forEach(card => card.remove());
    };

    const renderTableCards = (tableCards) => {
        elements.tableCards.innerHTML = '';

        if (!Array.isArray(tableCards)) return;

        tableCards.forEach(([card, playerIndex]) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'table-card';

            const cardEl = createCardElement(card);
            cardEl.classList.add('card-appear');
            wrapper.appendChild(cardEl);

            // Label
            const label = document.createElement('div');
            label.className = 'card-label';
            label.textContent = playerIndex === 0 ? 'Tu' : 'IA';
            wrapper.appendChild(label);

            elements.tableCards.appendChild(wrapper);
        });
    };

    const renderTrumpCard = (card) => {
        elements.trumpCard.innerHTML = '';
        if (card) {
            const cardEl = createCardElement(card);
            elements.trumpCard.appendChild(cardEl);
        }
    };

    const updateDeckCount = (count) => {
        elements.deckCount.textContent = count;
        elements.deck.style.display = count > 0 ? 'flex' : 'none';
    };

    const updatePlayerPoints = (points) => {
        elements.playerPoints.textContent = `${points} punti`;
    };

    const updateOpponentInfo = (name, points) => {
        elements.opponentName.textContent = name;
        elements.opponentPoints.textContent = `${points} punti`;
    };

    const showTurnMessage = (message, isThinking = false) => {
        elements.turnMessage.textContent = message;
        elements.turnMessage.className = 'turn-message' + (isThinking ? ' thinking' : '');
    };

    const showTrickResult = (message, duration = 2000) => {
        elements.trickResult.textContent = message;
        elements.trickResult.classList.remove('hidden');

        setTimeout(() => {
            elements.trickResult.classList.add('hidden');
        }, duration);
    };

    const displayGameResult = (result) => {
        let html = '';

        if (result.winner === 'Pareggio') {
            html = '<h3>Pareggio!</h3>';
        } else {
            html = `<h3>${result.winner} vince!</h3>`;
        }

        html += '<div class="scores">';
        for (const [name, points] of Object.entries(result.points || {})) {
            html += `
                <div class="score-item">
                    <div class="score-label">${name}</div>
                    <div class="score-value">${points}</div>
                </div>
            `;
        }
        html += '</div>';

        elements.resultContent.innerHTML = html;
        showGameResult();
    };

    const setPlayerName = (name) => {
        elements.playerNameDisplay.textContent = name;
    };

    return {
        init,
        showGameSetup,
        showGameBoard,
        showGameResult,
        updateGameInfo,
        renderPlayerHand,
        renderOpponentHand,
        revealOpponentCard,
        revealPlayerCard,
        removeRevealedCard,
        renderTableCards,
        renderTrumpCard,
        updateDeckCount,
        updatePlayerPoints,
        updateOpponentInfo,
        showTurnMessage,
        showTrickResult,
        displayGameResult,
        setPlayerName
    };
})();
