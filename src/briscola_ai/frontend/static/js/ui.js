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
        startGameButton: document.getElementById('start-game'),
        playerNameInput: document.getElementById('player-name-input'),
        aiAgentSelect: document.getElementById('ai-agent-select'),
        aiAgentDescription: document.getElementById('ai-agent-description'),
        aiAgentCommonNote: document.getElementById('ai-agent-common-note'),
        aiAgentCommonNoteText: document.getElementById('ai-agent-common-note-text'),
        aiModelGroup: document.getElementById('ai-model-group'),
        aiModelSelect: document.getElementById('ai-model-select'),
        aiModelDescription: document.getElementById('ai-model-description'),
        dataConsentGroup: document.getElementById('data-consent-group'),
        dataConsentCheckbox: document.getElementById('data-consent-checkbox'),
        dataConsentDescription: document.getElementById('data-consent-description'),
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

    // Metadati agenti IA caricati dal backend (source of truth: modulo Python).
    let aiAgentMetaByName = {};
    let aiAgentCommonNoteIt = '';

    // Modelli locali selezionabili (solo per `bc_model`).
    let aiModelMetaById = {};

    // Se true, richiediamo una checkbox esplicita prima di avviare la partita.
    let dataConsentRequired = false;

    const _updateAiAgentDescription = () => {
        const name = elements.aiAgentSelect?.value;
        const meta = name ? aiAgentMetaByName[name] : null;
        if (elements.aiAgentDescription) elements.aiAgentDescription.textContent = meta?.description_it || '';
        if (elements.aiAgentCommonNote && elements.aiAgentCommonNoteText) {
            const raw = (aiAgentCommonNoteIt || '').trim();
            const cleaned = raw.toLowerCase().startsWith('nota anti-cheat:')
                ? raw.slice('nota anti-cheat:'.length).trim()
                : raw;
            elements.aiAgentCommonNoteText.textContent = cleaned;
            elements.aiAgentCommonNote.classList.toggle('hidden', cleaned.length === 0);
        }
    };

    const _isBestAiModel = (model) => {
        const id = model?.id || '';
        const filename = model?.filename || '';
        // Campione consigliato: best_a2c_v3 (encoder v3). best_a2c (v2) resta selezionabile.
        return id === 'best_a2c_v3.npz' || filename === 'best_a2c_v3.npz';
    };

    const _formatAiModelOptionLabel = (model) => {
        const label = model?.label || model?.filename || model?.id || 'Modello locale';
        const filename = model?.filename || model?.id || '';
        const suffix = filename && filename !== label ? ` (${filename})` : '';
        const prefix = _isBestAiModel(model) ? 'Consigliato - ' : '';
        return `${prefix}${label}${suffix}`;
    };

    const _formatAiModelDescription = (model) => {
        if (!model) return '';

        const lines = [];
        if (_isBestAiModel(model)) {
            lines.push('Stato: best attuale consigliato');
        }

        const filename = model.filename || model.id;
        if (filename) {
            lines.push(`File: ${filename}`);
        }

        const guard = model.metadata?.inference_overkill_guard ?? model.metadata?.inference?.overkill_guard;
        if (typeof guard === 'boolean') {
            lines.push(`Guard anti-overkill: ${guard ? 'attivo' : 'non attivo'}`);
        }

        const desc = model.description_it || '';
        if (desc) {
            lines.push(desc);
        }
        return lines.join('\n');
    };

    const _updateConsentUi = () => {
        if (!elements.startGameButton) return;
        if (!dataConsentRequired) {
            elements.startGameButton.disabled = false;
            return;
        }
        const checked = elements.dataConsentCheckbox?.checked === true;
        elements.startGameButton.disabled = !checked;
    };

    const _updateAiModelUi = () => {
        const agentName = elements.aiAgentSelect?.value;
        const isBcModel = agentName === 'bc_model';

        if (elements.aiModelGroup) {
            elements.aiModelGroup.classList.toggle('hidden', !isBcModel);
        }

        if (!isBcModel) {
            if (elements.aiModelDescription) elements.aiModelDescription.textContent = '';
            return;
        }

        const modelId = elements.aiModelSelect?.value;
        const meta = modelId ? aiModelMetaById[modelId] : null;
        if (elements.aiModelDescription) {
            const desc = _formatAiModelDescription(meta);
            const compatible = meta?.is_compatible;
            const reason = meta?.compatibility_reason_it || '';
            if (compatible === false) {
                elements.aiModelDescription.textContent = `${desc}\n\nNON compatibile: ${reason}`.trim();
            } else {
                elements.aiModelDescription.textContent = desc;
            }
        }
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
            const modelId = elements.aiModelSelect?.value || null;
            const modelMeta = modelId ? aiModelMetaById[modelId] : null;
            callbacks.onStartGame?.({
                playerName: elements.playerNameInput.value || 'Giocatore',
                aiAgent: elements.aiAgentSelect?.value || 'random',
                aiAgentLabel: elements.aiAgentSelect?.selectedOptions?.[0]?.textContent || 'Random',
                aiModelId: modelId,
                aiModelLabel: elements.aiModelSelect?.selectedOptions?.[0]?.textContent || null,
                aiModelCompatible: modelMeta?.is_compatible === true,
                aiModelCompatibilityReasonIt: modelMeta?.compatibility_reason_it || null,
                consentToDataCollection: elements.dataConsentCheckbox?.checked === true,
            });
        });

        elements.aiAgentSelect?.addEventListener('change', () => {
            _updateAiAgentDescription();
            _updateAiModelUi();
        });
        elements.aiModelSelect?.addEventListener('change', _updateAiModelUi);
        elements.dataConsentCheckbox?.addEventListener('change', _updateConsentUi);

        elements.newGame.addEventListener('click', () => {
            callbacks.onNewGame?.();
        });
    };

    const setAiAgents = (catalog) => {
        const agents = Array.isArray(catalog) ? catalog : (catalog?.agents || []);
        aiAgentCommonNoteIt = Array.isArray(catalog) ? '' : (catalog?.common_note_it || '');

        if (!elements.aiAgentSelect || !Array.isArray(agents) || agents.length === 0) {
            _updateAiAgentDescription();
            _updateAiModelUi();
            return;
        }

        aiAgentMetaByName = {};
        agents.forEach((a) => {
            if (a?.name) aiAgentMetaByName[a.name] = a;
        });

        elements.aiAgentSelect.innerHTML = '';
        agents.forEach((a) => {
            if (!a?.name) return;
            const option = document.createElement('option');
            option.value = a.name;
            const available = a.available !== false;
            // Opzioni non disponibili (es. modello richiesto assente nel deploy) restano visibili
            // ma disabilitate: l'utente capisce che esistono ma non può selezionarle (niente errori).
            option.textContent = available ? (a.label || a.name) : `${a.label || a.name} (non disponibile)`;
            if (!available) {
                option.disabled = true;
                option.title = 'Modello richiesto non disponibile in questo deploy';
            }
            elements.aiAgentSelect.appendChild(option);
        });

        // Default: il "modello migliore" (bc_model + modello consigliato) se disponibile; in caso
        // contrario euristica v1; altrimenti il primo agente disponibile. Gli altri restano
        // selezionabili solo se l'utente vuole cambiare avversario.
        const isAvail = (name) => !!(name && aiAgentMetaByName[name] && aiAgentMetaByName[name].available !== false);
        const firstAvailable = agents.find((a) => a?.name && a.available !== false)?.name;
        let defaultAgent;
        if (isAvail('bc_model')) defaultAgent = 'bc_model';
        else if (isAvail('heuristic_v1')) defaultAgent = 'heuristic_v1';
        else defaultAgent = firstAvailable || agents[0]?.name || 'random';
        elements.aiAgentSelect.value = defaultAgent;
        _updateAiAgentDescription();
        _updateAiModelUi();
    };

    /**
     * Imposta la lista di modelli `.npz` disponibili (per l'agente `bc_model`).
     *
     * Payload atteso:
     * - `[{ id, label, description_it, ... }]`
     * - oppure `{ models: [...] }`
     */
    const setAiModels = (catalog) => {
        const models = Array.isArray(catalog) ? catalog : (catalog?.models || []);
        aiModelMetaById = {};

        if (!elements.aiModelSelect) {
            _updateAiModelUi();
            return;
        }

        elements.aiModelSelect.innerHTML = '';
        if (!Array.isArray(models) || models.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'Nessun modello trovato';
            option.disabled = true;
            option.selected = true;
            elements.aiModelSelect.appendChild(option);
            _updateAiModelUi();
            return;
        }

        const orderedModels = [...models].sort((a, b) => {
            if (_isBestAiModel(a) && !_isBestAiModel(b)) return -1;
            if (!_isBestAiModel(a) && _isBestAiModel(b)) return 1;
            return 0;
        });

        orderedModels.forEach((m) => {
            if (!m?.id) return;
            aiModelMetaById[m.id] = m;
            const option = document.createElement('option');
            option.value = m.id;
            option.textContent = _formatAiModelOptionLabel(m);
            if (m.is_compatible === false) {
                option.disabled = true;
                const reason = m.compatibility_reason_it ? ` (${m.compatibility_reason_it})` : '';
                option.textContent = `NON COMPATIBILE: ${option.textContent}${reason}`;
            }
            elements.aiModelSelect.appendChild(option);
        });

        // Default: best compatibile (se presente), altrimenti il primo modello compatibile.
        const firstCompatible = orderedModels.find((m) => m?.id && m.is_compatible !== false);
        if (firstCompatible?.id) elements.aiModelSelect.value = firstCompatible.id;
        else if (orderedModels[0]?.id) elements.aiModelSelect.value = orderedModels[0].id;

        _updateAiModelUi();
    };

    const setDataCollectionConsent = (payload) => {
        const required = payload?.required === true;
        const descriptionIt = payload?.description_it || '';
        dataConsentRequired = required;

        if (elements.dataConsentGroup) {
            elements.dataConsentGroup.classList.toggle('hidden', !required);
        }
        if (elements.dataConsentCheckbox) {
            elements.dataConsentCheckbox.checked = false;
        }
        if (elements.dataConsentDescription) {
            elements.dataConsentDescription.textContent = required
                ? (descriptionIt || 'Questa istanza sta raccogliendo dataset umano: serve il tuo consenso.')
                : '';
        }
        _updateConsentUi();
    };

    const showGameSetup = () => {
        elements.gameSetup.classList.remove('hidden');
        elements.gameBoard.classList.add('hidden');
        elements.gameResult.classList.add('hidden');
        // Se il consenso è richiesto, resettiamo la checkbox per rendere esplicita la scelta ad ogni partita.
        if (elements.dataConsentCheckbox) elements.dataConsentCheckbox.checked = false;
        _updateConsentUi();
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

    /**
     * Aggiorna informazioni "header" della partita (id + stato connessione).
     *
     * Nota:
     * - `connected` è utile come boolean base.
     * - `statusText`/`statusClass` permettono uno stato più granulare (es. "Riconnessione...").
     */
    const updateGameInfo = ({ gameId, connected, statusText, statusClass }) => {
        if (gameId) {
            elements.gameId.textContent = `ID: ${gameId.substring(0, 8)}...`;
        }
        if (connected !== undefined || statusText !== undefined || statusClass !== undefined) {
            const text = statusText !== undefined ? statusText : (connected ? 'Connesso' : 'Non connesso');
            elements.gameStatus.textContent = text;

            // Manteniamo l'id `game-status` e usiamo classi "stateful" per i colori.
            const classes = [];
            if (statusClass) classes.push(statusClass);
            else if (connected) classes.push('connected');
            elements.gameStatus.className = classes.join(' ');
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

        // Show/hide turn indicator (visibility mantiene lo spazio nel layout)
        elements.turnIndicator.style.visibility = isMyTurn ? 'visible' : 'hidden';
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
     * Ripristina lo stato visivo della mano del giocatore (rimuove highlight/disabled).
     *
     * Serve quando:
     * - la connessione WS cade durante un'azione
     * - la UI è in "hold" ma lo snapshot successivo non arriva (o arriva in ritardo)
     */
    const resetPlayerHandHighlights = () => {
        elements.playerHand.querySelectorAll('.card').forEach((cardEl) => {
            cardEl.classList.remove('revealed');
            cardEl.classList.remove('disabled');
        });
    };

    /**
     * Remove any revealed card from both player's and opponent's hands.
     * Use this when the card moves to the table.
     */
    const removeRevealedCard = () => {
        // Rimuovi carte evidenziate dalla mano avversario
        elements.opponentHand.querySelectorAll('.revealed').forEach(card => card.remove());
        // Rimuovi carte evidenziate dalla mano del giocatore
        elements.playerHand.querySelectorAll('.revealed').forEach(card => card.remove());
    };

    const renderTableCards = (tableCards) => {
        elements.tableCards.innerHTML = '';

        if (!Array.isArray(tableCards)) return;

        // Nuovo formato DTO: [{card, player_index}, ...]
        tableCards.forEach((item) => {
            const card = item.card;
            const playerIndex = item.player_index;

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

    const renderTrumpCard = (card, trumpSuit = null) => {
        elements.trumpCard.innerHTML = '';
        if (card) {
            const cardEl = createCardElement(card);
            elements.trumpCard.appendChild(cardEl);
            return;
        }

        // Placeholder sempre presente: mantiene stabile il layout anche quando il mazzo si esaurisce.
        // Quando non abbiamo (o non vogliamo mostrare) la carta, mostriamo comunque il seme di briscola (se noto).
        const suitNames = {
            clubs: 'Bastoni',
            cups: 'Coppe',
            coins: 'Denari',
            swords: 'Spade'
        };
        const label = document.createElement('div');
        label.className = 'trump-suit-indicator';
        if (trumpSuit) {
            label.textContent = `Briscola: ${suitNames[trumpSuit] || trumpSuit}`;
        } else {
            label.textContent = 'Briscola';
        }
        elements.trumpCard.appendChild(label);
    };

    const updateDeckCount = (count) => {
        const safeCount = Number.isFinite(count) ? count : 0;
        elements.deckCount.textContent = safeCount;

        // Manteniamo sempre visibile il placeholder del mazzo per evitare che l'area "tavolo"
        // cambi altezza quando il mazzo si esaurisce.
        elements.deck.style.display = 'flex';

        // Quando il mazzo è vuoto:
        // - non vogliamo più mostrare il retro della carta (sembra che ci sia ancora un mazzo)
        // - vogliamo un placeholder "vuoto" simile allo slot briscola.
        elements.deck.classList.toggle('deck-empty', safeCount <= 0);
        elements.deck.classList.toggle('card-back', safeCount > 0);
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
        elements.resultContent.replaceChildren();

        const title = document.createElement('h3');
        title.textContent = result.winner === 'Pareggio'
            ? 'Pareggio!'
            : `${result.winner || 'Risultato'} vince!`;
        elements.resultContent.appendChild(title);

        const scores = document.createElement('div');
        scores.className = 'scores';

        for (const [name, points] of Object.entries(result.points || {})) {
            const item = document.createElement('div');
            item.className = 'score-item';

            const label = document.createElement('div');
            label.className = 'score-label';
            label.textContent = name;

            const value = document.createElement('div');
            value.className = 'score-value';
            value.textContent = String(points);

            item.appendChild(label);
            item.appendChild(value);
            scores.appendChild(item);
        }

        elements.resultContent.appendChild(scores);
        showGameResult();
    };

    const setPlayerName = (name) => {
        elements.playerNameDisplay.textContent = name;
    };

    return {
        init,
        setAiAgents,
        setAiModels,
        setDataCollectionConsent,
        showGameSetup,
        showGameBoard,
        showGameResult,
        updateGameInfo,
        renderPlayerHand,
        renderOpponentHand,
        revealOpponentCard,
        revealPlayerCard,
        resetPlayerHandHighlights,
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
