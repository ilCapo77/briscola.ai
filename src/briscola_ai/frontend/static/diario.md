# Diario di bordo

*La storia vera di come un'IA ha imparato a giocare a Briscola: scelte, errori e svolte, senza dare per scontato che il lettore conosca reti neurali o training.*

## Prima di tutto: la Briscola

La [Briscola](https://it.wikipedia.org/wiki/Briscola) è uno dei giochi di carte più amati d'Italia: quaranta carte, un seme "di briscola" che batte gli altri, 120 punti in palio. Sopra i 60 si vince. Le regole si spiegano in pochi minuti; giocare bene richiede memoria, prudenza, tempismo e una buona dose di esperienza.

Per un progetto di intelligenza artificiale è un gioco interessante per due motivi. Primo: c'è **informazione nascosta**. Non vedi la mano dell'avversario e non sai che carte usciranno dal mazzo. Secondo: c'è **caso**. Una singola partita può raccontare poco, perché magari hai pescato benissimo o malissimo. Per capire se una strategia è davvero buona servono molte partite, giocate sugli stessi mazzi e misurate con un po' di statistica. È il contrario della sensazione da bar: "questa mano l'ho persa per colpa mia" a volte è vero, a volte no.

## Prologo <small>(gennaio 2026)</small>

Il progetto è partito con un obiettivo semplice da dire e lungo da realizzare: costruire da zero un'intelligenza artificiale capace di giocare bene a Briscola, e capire strada facendo quali metodi funzionano davvero. Non c'era solo da addestrare un modello. Servivano il motore delle regole, il tavolo online, gli strumenti per generare milioni di partite, i test per accorgersi subito quando qualcosa si rompe, e un modo serio per confrontare i giocatori. Tutto scritto in [Python](https://www.python.org/), il linguaggio più diffuso nel mondo dell'intelligenza artificiale: leggibile quasi come una lingua, comodo per sperimentare, e con un difetto che tornerà in questa storia — la lentezza, quando i calcoli si contano a miliardi.

All'inizio l'idea era quasi artigianale: costruire il banco da lavoro, poi gli attrezzi, poi un primo giocatore che impara copiando e sbagliando. L'immagine è comoda, ma la parte importante è più concreta: ogni scelta che sembrava promettente è stata messa alla prova con numeri, e parecchie sono state scartate.

C'è anche un elemento un po' particolare. Molto codice e molte decisioni sono nati dialogando con agenti di intelligenza artificiale come Claude, Codex e Gemini. Non come oracoli, ma come collaboratori da controllare: scrivono, propongono, sbagliano, vengono corretti dai test e dal maintainer. È un progetto che costruisce un'IA, costruito in parte insieme alle IA. Proprio per questo è stato utile tenerne traccia: su un progetto vero, con vincoli e regressioni, si capisce molto meglio cosa questi strumenti sanno fare.

Un'ultima nota prima di cominciare: il diario è ricostruito dalla storia scritta del repository. Commit, esperimenti, decisioni e i risultati con cui ogni campione si è guadagnato la promozione sono rimasti lì. Quando in un periodo non ci sono commit, non proveremo a inventare motivazioni: diremo semplicemente che non ci abbiamo lavorato.

## Capitolo 1 — Il tavolo da gioco <small>(gennaio 2026)</small>

Prima dell'intelligenza serviva un tavolo. Un agente che gioca solo nel terminale produce statistiche, ma le statistiche non ti dicono che effetto fa averlo come avversario. Per capirlo bisogna vederlo giocare, e il modo migliore è sedersi al tavolo contro di lui: carte in mano, briscola scoperta, prese, punti, tempi comprensibili. E poi, semplicemente, ogni tanto una partita ci si voleva giocare.

Il tavolo, in concreto, è un'applicazione web: una pagina che si apre nel browser, fatta di HTML e JavaScript, senza niente da installare. Dietro c'è un server scritto in Python (con un framework che si chiama [FastAPI](https://fastapi.tiangolo.com/)): è lui che custodisce la partita, applica le regole e fa giocare l'IA; la pagina si limita a mostrare le carte e a raccogliere le tue mosse.

Per quasi tutta questa storia il sito non era pubblico. Girava in locale, sul computer di chi lo stava costruendo. Sarebbe diventato pubblico solo a fine giugno, quando era abbastanza stabile da reggere utenti veri: ci arriveremo.

Il primo problema serio, curiosamente, non riguardò l'intelligenza: fu il ritmo della partita. Il modello decide in pochi millisecondi; se il tavolo mostrasse tutto a quella velocità, le carte apparirebbero e sparirebbero prima che tu abbia capito la presa. All'inizio sembrava naturale far comandare il browser: quando l'animazione finisce, chiama il server e fa giocare l'IA. Dopo tre giorni era chiaro che non reggeva bene: refresh nel momento sbagliato, doppi trigger, riconnessioni, stati intermedi difficili da spiegare. La soluzione adottata fu più semplice: il server avanza subito la partita, il frontend riceve gli eventi e li mostra con calma. La logica resta una sola; la pagina si occupa solo di raccontarla.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/01-timing-ui.md)*</small>

## Capitolo 2 — Le regole e il patto anti-cheat

Poi venne il motore delle regole. Può sembrare eccessivo dedicare tanto tempo a un gioco che molti imparano da bambini, ma qui ogni esperimento dipende da quello strato. Se il motore sbaglia una presa, o pesca nel momento sbagliato, tutti i risultati successivi diventano sospetti. Per questo il dominio del gioco è stato scritto in modo puro e deterministico. Il trucco è semplice: ogni partita nasce da un numero, il "seme", che decide il mescolamento del mazzo. Stesso seme, stesso mazzo, stesse pescate. È così che due giocatori possono essere confrontati sulle stesse identiche mani, o che una partita si può rigiocare tale e quale per capire cosa è successo.

Questa scelta non è solo pulizia tecnica. Permette di rifare un esperimento, confrontare due agenti sullo stesso mazzo, riprodurre una partita sospetta e capire dove è cambiato qualcosa. In un progetto in cui molti miglioramenti valgono mezzo punto a partita, la riproducibilità non è un lusso.

In quei giorni è nato anche il vincolo più importante del progetto: l'IA non deve sbirciare. Ogni agente riceve solo ciò che vedrebbe un giocatore leale: le proprie carte, il tavolo, la briscola, le carte già uscite. Non riceve la tua mano, non riceve l'ordine del mazzo, non riceve la prossima pescata. Sembra ovvio, ma per chi programma è una tentazione continua: un pezzetto di informazione nascosta semplificherebbe tanti problemi. Qui quella scorciatoia è vietata e controllata dai test. Se una modifica rompe il patto anti-cheat, i test la bloccano.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/02-dominio-anticheat.md)*</small>

## Capitolo 3 — Copiare, provare, fare sparring <small>(gennaio–febbraio 2026)</small>

Il primo maestro fu un giocatore scritto a mano: regole del tipo "se l'avversario ha giocato un carico e puoi prendere con una briscola bassa, fallo". Non è intelligenza nel senso moderno, ma gioca in modo dignitoso e dà una prima base. L'allievo, va detto una volta per tutte, è una **rete neurale**: in concreto, qualche decina di migliaia di numeri (i "pesi") che trasformano ciò che vede sul tavolo in una preferenza tra le carte che ha in mano. Imparare significa ritoccare quei numeri, un pochino alla volta, dopo ogni partita o ogni esempio. Il modello iniziò copiando migliaia di mosse del maestro di regole (in gergo si chiama *behavioral cloning*, clonazione del comportamento). Funzionò, però con un limite evidente: copiando un maestro ne erediti anche i difetti, e difficilmente lo superi.

Il passo successivo fu far giocare il modello per conto suo, milioni di volte, contro avversari congelati: è quello che nel progetto chiamiamo sparring, prendendo in prestito il termine dalla boxe (il nome tecnico è *apprendimento per rinforzo*: si impara dal premio e dalla punizione, non dagli esempi). Qui non gli viene detto "questa era la mossa giusta". Riceve il punteggio, prende botte, prova altre strade. Se butta una briscola preziosa per una presa povera, spesso lo paga più avanti. Dopo abbastanza partite, quel segnale debole diventa informazione utile.

La qualità dell'avversario conta parecchio. Se è troppo scarso, vinci quasi sempre e impari poco. Se è troppo forte, perdi senza capire perché. I salti migliori sono arrivati con avversari appena sopra il livello del modello, abbastanza forti da punirlo ma non così lontani da rendere il feedback incomprensibile.

Un caso rimasto nella storia del progetto è lo spreco delle briscole. Il modello dell'epoca usava briscole alte per vincere prese da due punti. Provammo a punirlo durante il training, due volte, con due penalità diverse. Peggiorò entrambe. Alla fine funzionò una soluzione più diretta: al momento della scelta, se sta per sprecare una briscola alta senza motivo, gli si fa usare la più economica che vince comunque. Non è elegante come "insegnarglielo", ma al tavolo funziona e non costa forza.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/03-prime-scuole-overkill.md)*</small>

## Capitolo 4 — Quattro mesi di pausa <small>(febbraio–giugno 2026)</small>

Poi, per quattro mesi, semplicemente non ci abbiamo lavorato. Succede. A giugno il lavoro riparte, e riparte da un collo di bottiglia molto concreto: la velocità.

## Capitolo 5 — La palestra veloce <small>(giugno 2026)</small>

Perché servono milioni di partite? Perché nella Briscola il risultato di una mano singola è troppo rumoroso. A volte vinci perché hai giocato bene, a volte perché il mazzo ti ha aiutato. Se vuoi distinguere le due cose devi ripetere lo stesso confronto molte volte, meglio ancora alternando i posti a tavola sullo stesso mazzo.

A giugno il motore veloce cambiò il ritmo del progetto. Ricordi il difetto di Python annunciato nel prologo? Eccolo. Python è un linguaggio interpretato: il computer legge le istruzioni una per una mentre le esegue, e nei cicli ripetuti miliardi di volte questo si paga. La versione leggibile restò la fonte di verità, ma per l'allenamento venne costruita una seconda implementazione pensata per la velocità. Due i cambiamenti. Primo, la rappresentazione: niente più "oggetti carta" con nome e seme, ma un numero da 0 a 39 per ogni carta, e mani, tavolo e mazzo ridotti a tabelle di numeri — il formato su cui i processori vanno più veloci. Secondo, la compilazione: con una libreria che si chiama [Numba](https://numba.pydata.org/), le funzioni critiche vengono tradotte in linguaggio macchina la prima volta che girano, e da lì in poi corrono come se fossero scritte in C, sfruttando anche tutti i core del processore in parallelo.

Il risultato pratico: circa quattordici volte più veloce, e cinque milioni di partite passate da qualche ora a un quarto d'ora. Una differenza così non migliora solo le prestazioni; cambia il modo in cui lavori. Un'idea costosa la provi raramente. Un'idea che costa quindici minuti la provi, la misuri e, se non va, la butti.

C'era però un vincolo: la versione veloce doveva giocare la stessa Briscola della versione canonica. Non "quasi". Stessa partita, stesso mazzo, stesse mosse, stessi esiti. Per questo il fast path è legato al dominio da test di parità. Ogni scorciatoia accettata deve dimostrare di non aver cambiato il gioco.

In quel periodo saltò fuori anche un errore molto concreto: un modello pesava 244 MB. Non era più intelligente degli altri. Dentro il file era finito l'intero registro delle metriche di allenamento. Ripulito, pesava 138 KB e giocava identico. Da lì in poi, quando un modello cambia peso, il peso viene guardato con sospetto.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/04-fast-numba.md)*</small>

## Intermezzo — Il sito va online <small>(23–25 giugno 2026)</small>

A fine giugno il tavolo uscì dal computer locale e diventò un sito pubblico. Sembra una formalità tecnica, ma in pratica cambiò parecchio. In locale c'è una sola istanza del server e la partita vive lì. Online possono esserci più repliche: la tua partita può essere letta da un processo e aggiornata da un altro. Servivano quindi uno store condiviso, lock per partita, WebSocket che funzionassero anche attraversando repliche diverse, modelli scaricati in modo verificabile e diagnostica per capire cosa fosse davvero attivo in produzione.

L'hosting scelto fu [FastAPI Cloud](https://fastapicloud.com/), il servizio pensato apposta per il framework con cui è scritto il server: si collega al repository e pubblica l'applicazione senza dover gestire macchine. E qui una nota di cui il maintainer va giustamente fiero: il sito gira praticamente **a costo zero**, incastrando i piani gratuiti dei vari servizi — l'hosting, il database per lo storico delle partite ([Neon](https://neon.tech/)), la memoria condivisa tra le repliche ([Redis](https://redis.io/)), e GitHub per distribuire i modelli. Un progetto didattico completo, online per il mondo, senza bolletta a fine mese.

Il lavoro durò tre giorni. Da allora l'indirizzo pubblico è [ai.briscola.dev](https://ai.briscola.dev).

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/10-deploy-cloud.md)*</small>

## Capitolo 6 — Un campione dopo l'altro, e un dubbio <small>(fine giugno 2026)</small>

Con la palestra veloce il progetto entrò in una routine di miglioramento. Attenzione a una cosa: quando qui si parla di "campioni" non sono giocatori diversi, ma **nuove varianti dello stesso modello** — la stessa rete neurale, con i pesi riaddestrati partendo dalla versione precedente, magari con più partite o qualche modifica alla ricetta. La variante nuova si allenava contro quella in carica, la batteva di poco nei test, e diventava il nuovo riferimento. Terza, quarta, quinta, sesta generazione. I progressi c'erano, ma diventavano più piccoli e più costosi.

A un certo punto emerse un dubbio serio: stavamo migliorando in assoluto, o stavamo solo imparando a battere il modello appena prima? Nei giochi non sempre la forza è transitiva. Battere A non garantisce di battere B, soprattutto se gli stili sono diversi. Da quel dubbio nacquero confronti più larghi: round-robin, intervalli di confidenza, test appaiati sugli stessi mazzi. E nacque anche una regola pratica: non avviare una nuova generazione solo perché la precedente è appena finita.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/05-catena-campioni.md)*</small>

## Capitolo 7 — Il solver del finale, e perché copiare non basta <small>(fine giugno 2026)</small>

La regola appena scritta — niente nuove generazioni per inerzia — aveva una conseguenza pratica: per continuare a migliorare servivano idee di natura diversa, non un'altra infornata di partite. La prima arrivò guardando il finale. Quando il mazzo è finito, l'informazione nascosta quasi sparisce: le carte sono quaranta, quelle uscite si conoscono, le proprie sono in mano, quindi quelle dell'avversario si deducono. In quel momento la Briscola diventa un problema molto più calcolabile.

Il solver fa proprio questo. Prende la posizione finale e prova tutte le continuazioni possibili fino all'ultima carta, assumendo che anche l'avversario giochi al meglio (la tecnica si chiama *minimax*: io scelgo il massimo per me, immaginando che lui scelga il minimo per me). Con poche carte per parte il numero di casi è piccolo per un computer. Alla fine non sceglie una mossa plausibile: sceglie quella dimostrata migliore per quella posizione.

Non si può usare lo stesso approccio da inizio partita per due motivi. Il primo è l'informazione nascosta: non sai cosa ha in mano l'avversario e non sai cosa verrà pescato. Il secondo è la dimensione: le ramificazioni esplodono. Il finale è il punto in cui il problema è piccolo e l'informazione è abbastanza completa.

Misurato sul campo, il solver del finale vale quasi due punti a partita senza cambiare il resto del giocatore. Da quel momento è diventato parte degli avversari forti del sito.

La seconda strada era più ambiziosa: far ragionare l'IA prima del finale, simulando possibili mondi nascosti, e poi far copiare quelle mosse al modello. Sembrava un buon modo per trasferire il ragionamento dentro la rete. Non funzionò. La rete imparò benissimo gli esempi visti, ma generalizzò male su quelli nuovi. In pratica aveva memorizzato il quaderno, non il criterio: è il fenomeno che nel settore si chiama *overfitting*, ed è una delle trappole più classiche di tutto il machine learning.

Quello che funzionò fu usare quel giocatore ragionante come avversario di allenamento. La settima generazione non copiò le sue mosse: ci giocò contro. E ne uscì con il salto più netto da mesi.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/06-search-endgame.md)*</small>

## Capitolo 8 — Tutte le ipotesi alla prova <small>(1–3 luglio 2026)</small>

A inizio luglio i progressi erano diventati piccoli e serviva capire dove intervenire. Le ipotesi erano tre: il modello non ricordava abbastanza della partita, era troppo piccolo, oppure il suo avversario di allenamento non aveva più molto da insegnargli.

Il metodo fu semplice: cambiare una cosa alla volta e confrontare con un gemello il più possibile identico, sugli stessi mazzi. Senza un controllo così, è troppo facile attribuire al modello un vantaggio che viene solo dal caso del mazzo.

I risultati furono misti. Aggiungere memoria delle prese passate — in concreto, 59 nuovi "sensori" in ingresso alla rete, che prima ne aveva 310 (le *feature*, in gergo): chi ha giocato cosa nelle ultime prese, chi ha tagliato, chi si è rifiutato di prendere — aiutò poco, ma in modo misurabile: era la prima prova davvero positiva del filone "più informazione". Un dettaglio carino del metodo: i nuovi sensori partono collegati con peso zero, così il modello modificato gioca all'inizio *esattamente* come l'originale, e ogni differenza che emerge dopo è merito o colpa loro. Aumentare la capacità della rete servì a poco o nulla. Aggiungere direttamente le probabilità sulle carte avversarie, invece, peggiorò il gioco: quelle probabilità erano ricavate dalle stesse informazioni che il modello vedeva già, e inserirle come input gli disturbava l'istinto invece di aiutarlo.

Anche il tentativo di usare un giudice di posizione (una *value network*: una seconda rete che guarda un tavolo e stima il risultato finale) per ragionare da inizio partita fallì. Non perché fosse implementato male, ma perché il valore di una posizione, quando restano tante carte nel mazzo, dipende troppo da pescate future che nessuno può sapere. La stima era troppo rumorosa per guidare bene le mosse.

Da quei giorni uscì comunque l'ottava generazione: memoria delle prese, rete più larga, altro sparring. Restò il modello principale del sito per un giorno solo. Il riassunto operativo era questo:

> «Il vincolo non è l'allievo. È il maestro.»

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/07-belief-exit.md)*</small>

## Capitolo 9 — L'avversario che simula i futuri <small>(3 luglio 2026)</small>

Una delle idee bocciate al capitolo precedente trovò posto altrove. Dire al modello, come input, "secondo me l'avversario ha queste carte" non aiutava. Usare la stessa stima per scegliere quali mondi simulare, invece, sì.

È così che nasce l'avversario più forte del sito. Per ogni mossa importante costruisce decine di mani avversarie compatibili con ciò che si è visto (le *determinizzazioni*: versioni "determinate" di un mondo incerto), dando più peso a quelle più probabili secondo il comportamento dell'avversario. Poi gioca quelle partite immaginarie fino in fondo e sceglie la mossa che rende meglio in media. Per chi gioca online è una pausa breve; per il programma sono centinaia di partite simulate fino in fondo (i *rollout*, nel gergo del settore).

Nel menu lo trovi come **"Modello locale + PIMC belief"**, un nome che ora puoi decodificare: PIMC sta per *Perfect Information Monte Carlo* — "immagina i mondi possibili e giocali" — e *belief* (credenza) è la stima su quali mondi siano più probabili, ricavata dal comportamento dell'avversario. Nelle valutazioni batte il campione base di quasi quattro punti a partita. La cosa interessante è che non nasce da un'idea nuova di zecca: nasce riciclando nel posto giusto un'idea che, data in pasto direttamente al modello, aveva fallito.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/08-pimc-belief.md)*</small>

## Capitolo 10 — La nonna aveva ragione

A questo punto venne una domanda molto pratica: le regole da giocatore esperto, quelle che si imparano al bar, vanno scritte a mano dentro l'IA? Non uscire coi carichi, non sprecare briscole su prese povere, non regalare figure.

Prima di aggiungere regole, le abbiamo misurate. Il risultato fu abbastanza sorprendente: il modello le rispettava già. Su alcuni comportamenti era perfino più prudente dell'euristica scritta a mano. Nessuno gli aveva detto esplicitamente quei precetti; li aveva trovati giocando, perché in media funzionano.

Resta un dettaglio: ogni tanto esce comunque con un carico. Non sembra per forza un errore. In molti casi è un carico giocato coperto, quando il modello sa di controllare le briscole. È una buona ragione per non trasformare ogni precetto umano in una regola rigida: a volte la regola è giusta, ma l'eccezione è parte del gioco.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/09-precetti-apertura.md)*</small>

## Capitolo 11 — I due allievi <small>(3–4 luglio 2026)</small>

Rimaneva una domanda aperta: si impara di più da un maestro molto forte, o giocando tantissimo contro avversari più vari? Invece di discuterne, il progetto fece due training paralleli dallo stesso punto di partenza: entrambi ripartivano dai pesi del campione in carica (un *warm start*, per non buttare via l'esperienza già accumulata).

Il primo seguiva una ricetta proposta dal maintainer: venti milioni di partite contro un mix di avversari. Per la maggior parte uno sparring forte, poi una quota contro se stesso e un 20% contro giocatori semplici, le vecchie euristiche, per non dimenticare come si vince contro chi gioca in modo più lineare. Il secondo seguiva l'idea opposta: molte meno partite, tutte contro il maestro più forte disponibile, quello con PIMC belief.

Dopo circa trenta ore complessive di calcolo, il risultato fu netto. Il modello allenato su volume e varietà vinse su tutta la linea: batté il campione precedente, superò il record storico e vinse lo scontro diretto con il gemello allenato dal maestro d'élite. Il secondo aveva imparato più in fretta per partita, quindi il maestro forte insegnava davvero qualcosa. Ma costava molto di più e, soprattutto, la dieta monotona gli aveva fatto perdere forza contro gli avversari semplici.

Il vincitore è diventato la nona generazione: il primo modello a migliorare contemporaneamente contro il predecessore e contro il riferimento storico. La lezione è meno poetica di quanto sembri: in questo regime, il chilometraggio e la varietà hanno battuto il maestro migliore. Il suo regno, però, durò poco.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/11-due-allievi.md)*</small>

## Capitolo 12 — Tutti gli avversari insieme <small>(4–5 luglio 2026)</small>

L'esperimento dei due allievi lasciava una domanda quasi ovvia, e fu il maintainer a farla: perché scegliere tra il maestro d'élite e la varietà, quando si possono avere entrambi? Nel frattempo era stato risolto anche l'ostacolo tecnico che lo impediva: il maestro che simula i futuri era stato riscritto nella forma veloce (lo stesso trattamento Numba del capitolo 5), abbastanza da poter reggere milioni di partite di allenamento.

Così partì il run più grande della storia del progetto: trenta milioni di partite contro una rosa di avversari che riuniva tutti — il maestro PIMC belief per un quarto del tempo, lo sparring quotidiano per un terzo, lo specchio, e un quarto abbondante di bar, la dose rialzata apposta. Circa ventisette ore di calcolo, sempre sullo stesso computer di casa.

Il risultato è il campione attuale, la decima generazione. Batte la nona di +0.66 punti a partita e porta il distacco dall'euristica storica a +20.5, il valore più alto mai misurato nel progetto: quasi due punti sopra il record che la nona aveva appena stabilito. Il merito di quel salto è soprattutto della quota bar: più partite contro i giocatori semplici hanno reso il modello più spietato proprio lì, senza togliergli nulla al vertice.

C'è anche il numero meno comodo, e va detto: la nona generazione aveva guadagnato +0.97 sulla precedente, la decima +0.66 — con più partite e avversari migliori. Ogni generazione compra un po' meno della precedente. Non è un fallimento: è la forma che ha il traguardo, in un gioco dove il mazzo decide comunque la sua parte. A un certo punto non resta molto da imparare; resta da giocare.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/12-giocatore-definitivo.md)*</small>

## Capitolo 13 — I giocatori veri <small>(7 luglio 2026)</small>

Il campione della decima generazione era online da qualche giorno quando è arrivata l'osservazione più utile del mese, da chi al sito ci stava semplicemente giocando: «è più forte degli altri, ma non è imbattibile». Verissimo. E per la prima volta avevamo abbastanza dati per dire *quanto*. Tra due giocatori di pari forza, una singola partita di Briscola balla di circa ventotto punti — su centoventi in palio — per puro effetto del mazzo. I divari di abilità veri sono molto più piccoli: il campione batte la generazione precedente di meno di un punto a partita, e perfino contro l'euristica scritta a mano, che stacca di venti punti, perde comunque una partita su quattro. Un giocatore imbattibile a Briscola non può esistere: prima ancora di toccare le carte, una decina di punti del risultato è già scritta nell'ordine del mazzo. La forza non decide la partita: inclina la moneta.

Poi siamo andati a guardare le partite vere, per la prima volta usando il registro di produzione come fonte: centoventitré partite umane complete, quaranta contro il campione, sette vinte dagli umani. Il diciassette per cento — esattamente quanto la statistica prometteva. Fin qui, nessuna notizia. La notizia era dentro quelle sette partite, e per capirla serve un pezzetto di strategia briscolistica.

A Briscola non c'è obbligo di seme: quando l'avversario apre una presa puoi rispondere con qualsiasi carta, e una carta di briscola — anche la più misera — batte qualunque carta degli altri semi. Giocarla su un seme altrui si chiama **tagliare**. Le briscole piccole (il due, il quattro, il cinque…) valgono zero punti in sé, ma sono la moneta migliore del gioco: se l'avversario apre un asso da undici punti e tu lo tagli con il due di briscola, hai comprato undici punti pagando zero. Non esiste scambio più conveniente. Il rovescio è che la tentazione di spenderle prima è continua — una presa qualsiasi da vincere, uno scarto comodo — e chi cede arriva a metà partita disarmato. I giocatori esperti le **conservano**: rinunciano a prese piccole pur di tenere la tagliola in mano, e aspettano. Prima o poi l'avversario un carico lo deve giocare.

Ecco cosa raccontavano le sette partite. Abbiamo rigiocato ogni mossa dell'IA facendola giudicare da una versione potenziata di lei stessa: verdetto, nessun errore. Ma il comportamento diceva altro: in quelle partite l'IA ha aperto la presa con un carico nove volte, e otto volte ha trovato la briscolina ad aspettarlo — centoundici punti trasferiti con questo solo meccanismo. Il giudice approvava quelle aperture perché è cieco al problema: nelle sue simulazioni l'avversario gioca come la sua famiglia, e nessuno in famiglia conserva le briscoline per tagliare. Nei suoi trenta milioni di partite di allenamento, aprire un asso a metà partita era ragionevolmente sicuro. I vincitori umani facevano l'opposto: aprivano con carte che non valgono niente, incassavano i carichi quando rispondevano — chi risponde, in due, non può essere tagliato da nessuno — e tenevano l'asso di briscola in mano oltre ogni consiglio del modello. Una partita su tutte lo dimostra: vinta 61 a 59 da un giocatore che aveva pescato un mazzo nettamente peggiore, con asso e tre di briscola dall'altra parte. Quella non era fortuna. Era tecnica.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/plans/audit-campo-2026-07-07.md)*</small>

## Capitolo 14 — La sonda che smentì sé stessa, e l'undicesima generazione <small>(7 luglio 2026)</small>

Lo stile dei vincitori era così riconoscibile che l'abbiamo scritto in un'euristica: apri liscio, incassa da secondo, conserva le briscoline per i carichi avversari, non sprecare mai briscole su prese povere. Quattro regole. Il risultato è il giocatore a regole più forte mai scritto nel progetto: dieci punti e mezzo sopra l'euristica storica. Ma non l'avevamo costruito per farlo giocare — l'avevamo costruito come **sonda**: se il campione fosse davvero vulnerabile a quello stile, contro la sonda avrebbe dovuto rendere meno di quanto la sua forza generale giustifica. È successo il contrario: il campione la gestisce benissimo. L'ipotesi dell'exploit, nata dall'analisi delle sette vittorie, si era appena smentita da sola — e la lezione ha un nome che in statistica si studia il primo giorno: *bias di selezione*. Avevamo analizzato solo le partite vinte dagli umani, mai le ventinove perse giocando allo stesso modo. Il buco dei carichi è reale, ma il campione recupera altrove quello che cede lì. La sonda resta comunque in squadra, con un mestiere nuovo: è il metro con cui misureremo se le prossime generazioni chiudono quel buco davvero.

Nel frattempo un'altra misura aveva dato l'esito opposto, e più promettente: il vantaggio del maestro che simula i futuri sul campione era rimasto **intatto** — la stessa forbice che aveva due generazioni prima. Trenta milioni di partite di allenamento non l'avevano consumato, perché quel vantaggio è strutturale: la ricerca media sui futuri possibili, una cosa che una rete reattiva non può replicare. E la versione agile della ricerca, sedici simulazioni invece di sessantaquattro, ne teneva quasi tutto il vantaggio a un sesto del costo.

Da lì l'ipotesi dell'undicesima generazione, semplice da dire: non più partite, ma **più maestro**. Quaranta per cento della rosa al maestro di ricerca — dose tolta al maestro precedente, che ormai insegnava poco — e appena cinque milioni di partite, sei volte meno del run monstre della decima. Tre ore e mezza di calcolo. Risultato: **+0.85** sulla decima, più di quanto la decima avesse strappato alla nona. La curva dei rendimenti decrescenti, che il capitolo scorso dava per destino, si è rialzata: contava la qualità della dose, non il volume. E il distacco dall'euristica storica ha toccato un nuovo massimo: +20.8.

Una sorpresa di contorno merita il suo paragrafo, perché chiude un cerchio aperto al capitolo 3. Ricordate la protesi anti-spreco, quella che impedisce al modello di buttare briscole alte su prese povere? Sull'undicesima generazione l'abbiamo misurata di nuovo: oggi è **dannosa** — mezzo punto a partita. Dopo milioni di partite contro maestri di ricerca, quelli che sembravano sprechi sono diventati scelte deliberate, e la stampella intralcia la gamba guarita. Il campione nuovo gioca senza. È la parabola di questo progetto in miniatura: ogni aiuto scritto a mano ha una scadenza, e la scopri solo misurando.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/plans/audit-campo-2026-07-07.md)*</small>

## Capitolo 15 — Il sito che dormiva <small>(7 luglio 2026)</small>

La stessa giornata aveva un secondo filone, molto più terra terra: perché il sito, a volte, ci mette venti secondi ad aprirsi? La risposta stava nei log della piattaforma: il piano gratuito spegne l'applicazione dopo appena novanta secondi senza visite, e la riaccende alla richiesta successiva. Il risveglio costava 18.8 secondi misurati — e per chi arriva, un sito che non risponde per venti secondi è un sito rotto. La radiografia disse che più di metà di quel tempo era la compilazione dei kernel veloci del capitolo 5: la palestra Numba, preziosa in allenamento, presentava il conto a ogni risveglio del tavolo.

Le prime due mosse furono di buon senso: spostare la compilazione in secondo piano, dietro alle richieste invece che davanti, e cuocere i modelli dentro l'immagine del server invece di riscaricarli a ogni risveglio. Il risveglio scese a 13.7 secondi — e lì si fermò, perché il resto è il prezzo della piattaforma, non nostro. Fu il maintainer ad alzare la posta: e se il sito facesse a meno di Numba del tutto? Il primo tentativo fu un disastro certificato: il solver del finale, richiamato migliaia di volte dentro le simulazioni della ricerca, in Python puro costava quattro volte tanto. Il secondo tentativo cambiò la domanda invece della risposta: perché risolvere lo stesso finale a ogni carta, quando si può risolverlo una volta sola e seguire la linea ottima fino in fondo? Con quel trucco — e con un solver compatto che era già in repo da settimane — il Python puro chiuse alla pari col kernel compilato: 16.6 millisecondi a mossa contro 17.0, esiti identici al bit, quarantasette megabyte di memoria in meno per replica, e più niente da compilare al risveglio. Oggi il tavolo online è Python puro da cima a fondo; i kernel veloci restano in palestra, dove i miliardi di calcoli servono davvero.

E mentre questo capitolo va in stampa, la dodicesima generazione sta correndo nella notte: dieci milioni di partite, e per la prima volta nella rosa c'è anche il castigatore — l'euristica nata dai vincitori umani, promossa da sonda ad avversario di allenamento. Domattina i numeri diranno se ha insegnato quello che le sette partite avevano mostrato.

## Epilogo

Se c'è un filo comune, non è la serie dei campioni promossi. È il numero di idee respinte: penalità che peggioravano il comportamento, quaderni di esempi imparati a memoria, reti più grandi che non servivano, stime di carte avversarie utili in un punto e dannose in un altro, giudici di posizione troppo rumorosi a inizio partita. Ogni bocciatura ha tolto un'ipotesi dal tavolo e ha reso più chiaro il passo successivo.

Alla fine il progetto resta didattico proprio per questo. Non perché ogni scelta sia stata giusta, ma perché quasi ogni scelta è stata misurata: una variabile alla volta, stessi mazzi quando possibile, margini di incertezza dichiarati, e abbastanza pazienza da buttare via un'idea quando i numeri non la sostengono.

*Il diario continua, e il futuro non è un segreto: le prossime mosse sono
scritte, come tutto il resto, in un [piano operativo pubblico](https://github.com/ilCapo77/briscola.ai/blob/master/PLAN.md).
La progressione dei campioni — con i risultati misurati con cui ognuno si è guadagnato il posto — è in un
[report scaricabile (Excel)](https://github.com/ilCapo77/briscola.ai/raw/master/docs/reports/model_progress.xlsx); il
[codice e la storia completa](https://github.com/ilCapo77/briscola.ai) sono su GitHub.*
