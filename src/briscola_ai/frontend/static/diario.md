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

<h2 id="patto-anti-cheat">Capitolo 2 — Le regole e il patto anti-cheat</h2>

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

## Capitolo 13 — Cosa insegnano le prime partite umane <small>(7 luglio 2026)</small>

Il campione della decima generazione era online da qualche giorno quando è arrivata una frase molto normale, detta da chi stava giocando davvero: «è più forte degli altri, ma non è imbattibile». Era il commento giusto. In laboratorio un modello si misura su centomila partite; sul sito, invece, una persona vede una mano alla volta. E una mano di Briscola, da sola, è rumorosa: tra due giocatori di pari forza la differenza finale balla di circa ventotto punti su centoventi solo per effetto del mazzo.

Questo numero cambia il modo di leggere tutto il resto. Quando diciamo che una generazione batte la precedente di meno di un punto a partita, quel vantaggio esiste solo sulla media di moltissime partite; in una partita singola non lo puoi "sentire". Perfino contro l'euristica scritta a mano, che prende venti punti di distacco, il campione perde comunque una partita su quattro o cinque. Briscola non permette un giocatore imbattibile. La bravura sposta le probabilità, non cancella il mazzo.

Poi siamo andati a guardare le partite vere, usando per la prima volta il registro di produzione come materiale di analisi: centoventitré partite umane complete, quaranta contro il campione, sette vinte dagli umani. Il 17,5%. Più o meno quello che ci aspettavamo. La parte interessante, quindi, non era "l'IA perde". Era: *come* perde quando perde.

Per capirlo serve ricordare una cosa semplice del gioco. A Briscola non c'è obbligo di seme: se l'avversario apre una presa puoi rispondere con qualsiasi carta. Una briscola piccola, anche una carta da zero punti, batte qualunque carta degli altri semi. Questa mossa si chiama **tagliare**. Se l'avversario apre con un asso da undici punti e tu lo tagli con il due di briscola, hai preso undici punti pagando zero. Per questo i giocatori prudenti conservano le briscole piccole: sembrano carte povere, ma sono il modo più economico per rubare i carichi avversari.

Nelle sette vittorie umane il pattern era chiaro. Abbiamo rigiocato ogni mossa dell'IA facendola giudicare da una versione più pesante dello stesso sistema, e il giudice non ha trovato errori netti. Ma quel giudice ragiona dentro la stessa cultura del modello: nelle sue simulazioni gli avversari non tengono le briscoline in mano per punire un carico guidato. Gli umani, invece, lo facevano. In quelle partite l'IA ha aperto con un carico nove volte; otto sono finite tagliate da una briscola piccola, per circa centoundici punti trasferiti. Una partita è diventata il caso da tenere sul tavolo: 61 a 59 per l'umano nonostante un mazzo peggiore, con asso e tre di briscola finiti all'IA. Non prova da sola un exploit generale, ma mostra bene il tipo di comportamento da indagare.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/13-fortuna-e-campo.md)*</small>

## Capitolo 14 — Un test sullo stile umano e la nascita della v11 <small>(7 luglio 2026)</small>

Lo stile dei vincitori umani era abbastanza preciso da poterlo trasformare in un giocatore a regole: apri con carte lisce, incassa i carichi quando rispondi, conserva le briscole piccole per tagliare, non sprecarle su prese povere. Non era pensato come nuovo avversario del sito. Era uno strumento di misura, una **sonda**: se il campione soffriva davvero quello stile, contro questa euristica avrebbe dovuto andare peggio del previsto.

La sonda era forte, più forte di qualunque altra euristica scritta nel progetto. Ma contro il campione non trovò il varco che cercavamo. Il campione la batteva meglio di quanto avrebbe suggerito un conto semplice di forza relativa. Questo non cancella il problema visto nelle sette vittorie: i carichi guidati vengono ancora puniti. Però corregge la diagnosi. Avevamo guardato solo le partite vinte dagli umani, non quelle perse da persone che magari giocavano in modo simile. È il classico *bias di selezione*: se osservi solo i casi riusciti, lo stile sembra più decisivo di quanto sia davvero. Da quel momento la sonda ha cambiato mestiere. Non è più "l'exploit umano"; è il termometro per capire se una generazione futura migliora proprio su quel fianco.

Quasi nello stesso momento è arrivata una misura molto più incoraggiante. Il maestro che simula futuri possibili, quello basato su PIMC belief, aveva ancora lo stesso vantaggio sul campione che aveva due generazioni prima. Trenta milioni di partite di allenamento non lo avevano assorbito. Il motivo è abbastanza intuitivo: una rete reattiva sceglie da ciò che vede ora, mentre la ricerca prova molti futuri compatibili con le carte nascoste e fa una media. Sono due modi diversi di decidere. La versione più leggera della ricerca, con sedici simulazioni invece di sessantaquattro, conservava quasi tutto il vantaggio costando molto meno.

La ricetta dell'undicesima generazione nacque da lì: meno volume, più dose del maestro giusto. Cinque milioni di partite, sei volte meno della decima generazione, ma il 40% degli avversari era il maestro di ricerca leggero. Dopo tre ore e mezza di calcolo, il risultato fu netto: **+0.85** sulla decima, più di quanto la decima avesse guadagnato sulla nona. La morale operativa non è "servono sempre più partite"; è più precisa: serve capire quale avversario ha ancora qualcosa da insegnare.

Prima di promuoverla abbiamo rimisurato anche un vecchio aiuto scritto a mano: la protezione anti-spreco che impediva al modello di usare briscole alte su prese povere. Molte generazioni prima, al capitolo 3, era stata utile. Sulla v11 costava mezzo punto a partita. Alcune mosse che sembravano sprechi, ormai, erano scelte che il modello sapeva motivare col contesto. La regola è stata tolta. È un buon promemoria per tutto il progetto: una correzione manuale non resta vera per sempre; va rimisurata quando il giocatore cambia.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/14-sonda-e-dose-shift.md)*</small>

## Capitolo 15 — Perché il sito si svegliava lentamente <small>(7 luglio 2026)</small>

La stessa giornata aveva un secondo filone, molto più terra terra: perché il sito, a volte, ci metteva venti secondi ad aprirsi? La risposta era nei log della piattaforma. Il piano gratuito spegne l'applicazione dopo circa novanta secondi senza visite e la riaccende alla richiesta successiva. Per chi arriva sul sito, però, il dettaglio tecnico non conta: se una pagina resta muta per venti secondi, sembra rotta.

La prima radiografia diceva che il risveglio costava 18,8 secondi. Più di metà era compilazione Numba, la tecnologia che nel capitolo 5 aveva reso possibile allenare milioni di partite in tempi umani. In allenamento è preziosa; sul sito pubblico, con lo spegnimento automatico, diventava una tassa pagata al primo visitatore dopo ogni pausa.

Abbiamo fatto prima le cose ovvie: spostare la compilazione in secondo piano e mettere i modelli direttamente nell'immagine del server, invece di scaricarli a ogni risveglio. Il primo tempo di risposta è sceso a 13,7 secondi. Poi si è fermato lì. A quel punto il costo non era più quasi nostro: erano avvio del container, boot dell'applicazione e controlli della piattaforma. Per questo nel sito è comparso anche un avviso più onesto: quando il server si sta svegliando, il giocatore deve saperlo.

Restava una domanda tecnica: si poteva togliere Numba dal runtime web, lasciandolo solo alla palestra di training? Il primo tentativo non andò bene. Il solver del finale, chiamato molte volte dentro le simulazioni della ricerca, in Python puro rendeva ogni mossa quattro volte più lenta. Il secondo tentativo funzionò perché cambiò il punto in cui si calcola: invece di risolvere lo stesso finale per ogni carta candidata, si risolve una volta sola e si segue la linea migliore fino in fondo. Con quella modifica il tavolo online è diventato Python puro: 16,6 millisecondi a mossa contro 17,0 del kernel compilato, stessi esiti a parità di seed, quarantasette megabyte in meno per replica e nessuna compilazione al risveglio. I kernel veloci restano dove servono davvero: negli esperimenti lunghi.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/15-zero-numba.md)*</small>

## Capitolo 16 — Perché v12 non basta: serve riconoscere lo stile <small>(8 luglio 2026)</small>

Dopo la promozione della v11 restava una tentazione naturale: prendere la sonda nata dalle vittorie umane, metterla nel gruppo degli avversari di allenamento, e vedere se il modello imparava a non offrire più carichi ai conservatori di briscole. È diventata la dodicesima generazione: dieci milioni di partite, con il nuovo avversario presente nel 12% degli incontri di training.

Il risultato è stato utile proprio perché deludente. La v12 giocava forte quasi quanto la v11, ma non meglio: +0,11 punti, troppo poco per chiamarlo progresso. Soprattutto, non aveva cambiato abitudine. Contro il giocatore conservatore guidava carichi praticamente con la stessa frequenza della v11 e li perdeva nella stessa proporzione. La punizione c'era, ma era diluita: contro quel 12% di partite una certa apertura costava, contro molti altri avversari spesso pagava. Una rete che non riconosce lo stile dell'avversario fa la media, e la media non cambia.

Per evitare di inseguire un altro run a vuoto, abbiamo fatto una cosa meno appariscente: una "pagella della nonna". Lo script non misura se il modello vince; misura abitudini concrete, quelle che un giocatore umano nominerebbe al tavolo. Apre liscio? Conserva le briscole piccole? Regala punti quando perde una presa? Scarta dal seme corto per prepararsi a tagliare? Tiene l'asso di briscola per il finale?

La v11 non è una caricatura. Su molte cose è solida: quando perde una presa regala pochissimo, si "sbianca" spesso dal seme corto e tende a tenere l'asso di briscola fino alle ultime prese. Però il profilo cambia pochissimo contro avversari diversi. E c'è una sorpresa grossa: quando ha tante briscole in mano, non esce mai a briscola bassa per far consumare quelle dell'altro. È il contrario della regola tradizionale "cavare le briscole". Può darsi che nel 1v1 abbia ragione il modello, ma è una domanda da misurare, non da risolvere a intuito.

Il prossimo passo, quindi, non è un'altra notte di training. È raccogliere altre partite umane e dare al modello qualche indizio pubblico sullo stile dell'avversario: quante volte taglia i carichi, quanto apre liscio, quanto conserva o spreca briscole. Non è barare; sono riassunti della storia visibile della partita. Se funzionerà, il segnale dovrà essere preciso: meno carichi regalati contro chi conserva le briscole, non prudenza generica contro tutti.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/16-ascolto-e-stile.md)*</small>

## Capitolo 17 — Stessa forza, comportamento migliore <small>(9 luglio 2026)</small>

La v12 ci aveva lasciato con una domanda ancora aperta. Avevamo visto giocatori umani vincere conservando le briscole, tagliando i carichi e punendo certe aperture dell'IA. Avevamo provato a mettere quello stile nel training, ma il risultato era stato quasi nullo: la v12 giocava forte come la v11, non meglio, e non cambiava davvero abitudine. Restava quindi il dubbio più interessante: quei comportamenti erano errori del modello, oppure ci sembravano errori perché li guardavamo dalle partite perse?

Il caso più evidente erano i carichi guidati. In alcune sconfitte l'IA usciva con un asso o un tre non di briscola, l'umano tagliava, e la presa sembrava raccontare da sola il problema. Ma una mano singola è una cattiva testimone. Come sempre, l'impressione è diventata un'ipotesi da misurare: prima sulle partite vinte dagli umani, poi sul tasso base di tutte le partite disponibili.

Il risultato ha ridimensionato l'aneddoto. I carichi guidati tagliati dagli umani erano reali, ma non erano un marchio di sconfitta: capitavano spesso anche nelle partite che l'IA vinceva. Peggio ancora per l'ipotesi del "difetto": quando abbiamo provato a impedirli con un guard, il modello perdeva punti. Quella pista si è chiusa lì. Comportamento visibile, sì; difetto sfruttabile, no.

Lo stesso è successo con altri sospetti. L'asso di briscola veniva forse giocato troppo presto? No: quasi sempre usciva tardi e catturava punti veri. La "cavata", cioè aprire in briscola quando se ne hanno molte in mano, sembrava aggressiva? Messa alla prova contro l'alternativa, era una scelta buona: impedirla faceva crollare il punteggio. Anche l'idea di allargare la finestra PIMC da 8 a 10 non ha retto: più ragionamento, in quel punto, non significava gioco migliore.

La lezione non era molto comoda, ma era pulita: molte mosse che sembravano brutte erano semplicemente difficili da valutare a occhio. In una partita singola vedi il carico tagliato e pensi "errore". Su diecimila partite scopri che quel sacrificio, in media, fa parte di un piano che funziona.

Restava però un comportamento diverso: l'overkill di briscola. Da secondo di mano, su un piatto povero, il modello a volte vinceva con una briscola più alta del necessario, anche quando ne aveva una più economica che avrebbe preso comunque. Questo caso era più concreto. Non stiamo dicendo "non prendere", ma "prendi con la carta meno costosa". E soprattutto era abbastanza frequente da dare al training un segnale vero.

Così è nata la tredicesima generazione. Non un nuovo grande cambio di architettura, non nuove feature, non un guard scritto a mano al momento della mossa. Solo una penalità morbida durante l'allenamento: se vinci una presa povera sprecando una briscola troppo alta, paghi un piccolo costo proporzionale allo spreco. La forza della penalità è stata scelta dopo varie prove: abbastanza forte da cambiare l'abitudine, non così forte da trasformarsi in una regola rigida.

Il risultato è esattamente il tipo di progresso che bisogna raccontare con onestà. La v13 non è più forte della v11 in modo significativo. Sulla sola policy fa -0,03 punti a partita, con intervallo di confidenza da -0,38 a +0,32. Nel confronto che conta per il sito, cioè il default reale con PIMC belief 16x8, fa +0,14, con intervallo da -0,20 a +0,47. In entrambi i casi siamo dentro il rumore statistico.

Contro gli avversari di controllo non si rompe nulla: nel default PIMC 16x8 fa +16,30 contro il conservatore di briscole, contro +16,06 della v11, e +22,40 contro l'euristica storica, contro +22,48 della v11. Differenze piccole, non una nuova soglia di forza.

Però cambia il comportamento che volevamo cambiare. L'overkill di briscola su piatti poveri scende da circa 28-31% a circa 6-8%. In pratica, il modello continua a giocare forte come prima, ma molto più spesso prende con la briscola giusta invece di pagare più del necessario.

Per questo la v13 diventa il nuovo default. Non perché sia "più intelligente" in senso assoluto, ma perché è un giocatore più pulito. La frase giusta è: stessa forza, comportamento migliore.

È una promozione diversa dalle precedenti. Le prime generazioni cercavano punti. Questa corregge un'abitudine senza comprare forza. In un progetto didattico è comunque un risultato importante: dopo molte ipotesi bocciate, abbiamo trovato un caso in cui il reward shaping non serve a vincere di più, ma a insegnare al modello a vincere nello stesso modo con meno spreco.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/17-stessa-forza-comportamento-migliore.md)*</small>

## Capitolo 18 — I semi non hanno nome <small>(11 luglio 2026)</small>

Dopo la v13 abbiamo resistito alla tentazione di lanciare subito un altro allenamento. Il modello era arrivato a un punto in cui aggiungere partite, neuroni o regole senza sapere dove guardare rischiava di produrre soltanto un'altra variante quasi uguale. Serviva prima una domanda capace di isolare un difetto preciso.

La domanda è semplice da raccontare. Prendiamo una posizione di Briscola e cambiamo i nomi dei quattro semi dappertutto: la mano, la briscola, il tavolo e tutte le carte già viste. I denari diventano coppe, le coppe diventano spade e così via. Non abbiamo cambiato la partita; abbiamo soltanto ristampato gli stessi simboli. La mossa del modello dovrebbe cambiare nome nello stesso modo, ma restare la stessa scelta strategica.

Abbiamo provato tutte le 24 rinomine possibili su 4.096 decisioni della v13, distribuite fra quattro avversari e quattro fasi della partita. Le mosse obbligate sono state escluse: quando resta una sola carta, qualunque modello sembra perfettamente coerente. Anche il controllo più importante ha funzionato: senza rinominare nulla, il risultato torna identico fino all'ultimo numero.

La sorpresa è grande. Nel 18,19% dei confronti la v13 sceglie una carta diversa soltanto perché abbiamo cambiato i nomi dei semi. E in poco più della metà delle posizioni basta almeno una delle 23 rinomine non banali per far cambiare la decisione. Non sono esitazioni sul filo del pareggio: nessuna delle 98.304 versioni esaminate aveva le prime due carte quasi pari secondo il modello.

Il fenomeno attraversa tutta la partita. È un po' più forte quando l'IA apre la presa e quando ha tre carte fra cui scegliere, ma non sparisce nel finale, contro lo specchio o contro le euristiche. In alcuni casi le probabilità restano quasi uguali; in altri il modello passa con decisione quasi totale da una carta a un'altra.

Come può succedere? Per la rete i quattro semi occupano blocchi numerici distinti. Durante l'allenamento ha visto tutti i semi, ma nessuno le ha imposto che i loro nomi fossero intercambiabili. Può quindi aver conservato associazioni accidentali: non una regola della Briscola, ma una preferenza legata alla posizione in cui un seme finisce nel vettore.

Questo risultato non dice ancora che abbiamo trovato una mossa capace di farla vincere di più. Due carte diverse possono valere quasi lo stesso nella partita reale, anche se il modello è molto sicuro. Dice però che abbiamo finalmente una leva concreta e misurabile.

Il prossimo esperimento sarà insegnare la simmetria durante l'allenamento. Nello stesso aggiornamento, alla traiettoria originale affiancheremo una copia con i semi rinominati e con tutte le mosse rinominate nello stesso modo. La coppia è importante: i mazzi casuali mostrano già tutti i semi con la stessa frequenza, quindi cambiare soltanto i nomi di un singolo esempio non aggiungerebbe una regola nuova. Non è un trucco da applicare sul sito e non costa tempo quando si gioca. La prova sarà superata solo se i cambi di scelta diminuiranno senza perdere punti contro gli avversari di controllo.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/18-i-semi-non-hanno-nome.md)*</small>

## Capitolo 19 — La copia non basta <small>(11 luglio 2026)</small>

La prova proposta nel capitolo precedente sembrava quasi ovvia: se il modello si lascia influenzare dai nomi dei semi, mostriamogli due volte la stessa partita, prima con i nomi originali e poi con tutti i semi rinominati nello stesso modo. La mossa e il risultato restano gli stessi. In teoria, la rete dovrebbe imparare a ignorare le etichette.

Prima di allenare abbiamo costruito una trasformazione controllata. Non cambia soltanto le carte in mano: rinomina anche briscola, tavolo, carte viste, memoria delle prese e le quaranta possibili mosse. I test confrontano questa scorciatoia numerica con la trasformazione completa della partita per tutte le 24 rinomine. Abbiamo verificato anche che, con l'opzione spenta, il trainer produca esattamente gli stessi pesi e le stesse informazioni di prima.

Poi abbiamo creato sei copie della v13. Tre hanno continuato l'allenamento normale e tre hanno ricevuto, a ogni aggiornamento, anche la versione rinominata. Ogni coppia è partita dallo stesso modello, ha giocato gli stessi 10.000 mazzi e ha affrontato lo stesso gruppo di avversari. In questo modo l'unica differenza era la nuova lezione sui semi.

Il risultato ha segno opposto a quello sperato. Nei tre controlli il cambio di carta dovuto ai nomi dei semi resta in media al 18,32%. Nei tre modelli con la copia sale al 18,84%. Anche la distanza fra le probabilità aumenta. Nei confronti diretti i modelli con la copia perdono in media 0,15 punti a partita rispetto ai loro controlli: poco, ma non c'è nessun guadagno che compensi il peggioramento della misura principale.

Perché una lezione apparentemente corretta può fallire? Durante l'apprendimento per rinforzo una mossa non è soltanto un'etichetta scritta su un esempio: è una scelta che il giocatore ha davvero estratto dalle proprie probabilità. La mossa originale è stata scelta guardando la posizione originale. La sua copia rinominata, invece, non è stata scelta di nuovo dalla rete nella seconda posizione. Finché le due versioni della rete ragionano in modo diverso, trattarle come se avessero prodotto entrambe la stessa esperienza può aggiungere confusione.

Diecimila partite sono uno screening, non una sentenza matematica su qualunque allenamento futuro. Ma lo screening serve proprio a questo: evitare milioni di partite quando tutti i primi segnali vanno nella direzione sbagliata. Nessun nuovo modello viene promosso e la v13 resta invariata.

La trasformazione costruita non è sprecata. Il prossimo tentativo sarà più diretto: il modello guarderà la posizione originale e quella rinominata, e riceverà una piccola penalità quando le due risposte non coincidono dopo aver riallineato i nomi delle carte. Questa è una *consistency loss*, cioè un costo che chiede esplicitamente coerenza. L'allenamento normale continuerà a usare soltanto le mosse davvero giocate.

La prima prova di questa seconda strada ha finalmente il segno giusto. Con tre intensità crescenti e tre allenamenti per ciascuna, il cambiamento di carta scende poco alla volta fino al 15,64%, contro il 18,32% dei controlli. La distanza fra le risposte diminuisce ancora più chiaramente. Nei confronti diretti contro la v13 la forza resta invece indistinguibile: la differenza media è -0,08 punti a partita e i margini di incertezza comprendono sempre la parità.

Non è ancora una soluzione e non nasce un nuovo campione. È però la prima prova che chiedere esplicitamente coerenza corregge davvero una parte del difetto senza pagare una perdita evidente nel gioco. Il passo successivo sarà un allenamento intermedio, con soste programmate per controllare se il miglioramento continua oppure si ferma.

Le soste hanno dato una risposta netta. A 30.000 partite il cambiamento di carta resta quasi fermo al 15,46% e la forza è ancora pari alla v13. A 50.000, invece, il cambiamento risale al 16,47% e tutti e tre gli allenamenti diventano più deboli, perdendo in media 0,77 punti a partita. Le probabilità delle versioni rinominate sono più vicine, ma il modello è anche meno deciso: la distanza fra la prima e la seconda carta si restringe, e basta una differenza più piccola per scambiarle.

Anche questa strada quindi si ferma prima della promozione. Il prossimo tentativo non dovrà chiedere soltanto probabilità simili: dovrà proteggere direttamente la carta scelta e il vantaggio che la separa dalla seconda scelta.

La penalità sul margine riesce a conservare quella sicurezza e abbassa il cambiamento di carta fino al 14,42%, ma poi si ferma: renderla dieci volte più forte non produce un altro passo avanti. La forza contro la v13 resta indistinguibile, quindi neppure questo esperimento crea un nuovo campione.

A questo punto smettiamo temporaneamente di insegnare. Il prossimo controllo prenderà la v13 così com'è, la farà rispondere a tutte le 24 versioni della stessa posizione e medierà le risposte riallineate. Il risultato sarà perfettamente indipendente dai nomi dei semi. Potremo così chiedere direttamente se eliminare il difetto fa davvero giocare meglio, prima di investire in un'altra architettura.

<small>*Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/19-la-copia-non-basta.md)*</small>

## Capitolo 20 — Ventiquattro pareri, una sola mossa <small>(11 luglio 2026)</small>

Dopo tre tentativi di insegnare la coerenza, la domanda più importante era ancora aperta. Il modello cambiava carta quando cambiavamo soltanto i nomi dei semi, ma non sapevamo se questo gli facesse davvero perdere partite. Due mosse diverse possono avere lo stesso valore, e correggere un difetto interno non garantisce un giocatore più forte.

Questa volta non abbiamo allenato nulla. Abbiamo preso la v13 così com'è e, per ogni decisione, le abbiamo mostrato tutte le 24 ristampe possibili della stessa posizione. Ogni risposta è stata riportata ai nomi originali delle carte; poi abbiamo fatto la media e scelto la carta col valore più alto. È come chiedere ventiquattro pareri allo stesso giocatore dopo aver cambiato soltanto le etichette sul tavolo.

Per costruzione, il risultato non può più dipendere dai nomi dei semi. Il controllo su 160 posizioni reali e 3.680 rinomine non banali ha trovato zero cambi di carta. E il costo è molto più piccolo di quanto suggerisca il numero ventiquattro: le copie entrano nella rete tutte insieme, in un *batch*, quindi la latenza media passa da circa 0,051 a 0,074 millisecondi. Una volta e mezza, non ventiquattro.

Il confronto diretto ha finalmente separato il sintomo dalla causa. Su diecimila partite con gli stessi mazzi e i posti scambiati, la versione simmetrica batte la v13 originale di 0,90 punti a partita. Il margine d'incertezza va da +0,47 a +1,33: questa volta la parità resta fuori. Anche contro le due euristiche di controllo il vantaggio cresce di circa sette decimi.

Non abbiamo ricomprato quei punti tornando a sprecare briscole. L'overkill sui piatti poveri, il comportamento corretto dalla v13, scende ancora: dall'8,0% al 3,9% nel controllo contro l'euristica storica. Le volte in cui usa una briscola anche se una carta normale avrebbe già vinto scendono da 99 a 41.

Ora sappiamo qualcosa che prima mancava: le associazioni accidentali con i nomi dei semi non erano soltanto disordine nella testa del modello. Gli costavano davvero forza.

La versione a ventiquattro pareri non diventa subito il giocatore del sito. Il default usa la ricerca PIMC e richiama la policy molte volte dentro le sue simulazioni; quel costo va studiato nel contesto giusto. Il prossimo passo sarà invece usare la risposta media come maestro e insegnarla a una sola rete. Il termine tecnico è *distillazione*: trasferire il comportamento di un sistema più costoso in un modello più compatto. Se funziona, una sola risposta conserverà ciò che oggi otteniamo chiedendone ventiquattro.

<small>*Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/20-ventiquattro-pareri.md)*</small>

## Capitolo 21 — Una voce sola <small>(11 luglio 2026)</small>

Il consiglio a ventiquattro aveva risposto alla domanda scientifica, ma non era ancora la soluzione più semplice da mettere in campo. Per ogni mossa chiedeva alla v13 di guardare tutte le ristampe della posizione e poi mediava le risposte. Funzionava e costava meno del previsto, ma il giocatore del sito usa quella policy molte volte dentro le simulazioni PIMC. Portare lì ventiquattro pareri per ogni scelta interna sarebbe un cambiamento più grande.

Abbiamo quindi provato a trasferire il risultato in una sola rete. Il metodo si chiama *distillazione*: un sistema più costoso fa da maestro, un modello compatto osserva le sue risposte e impara a riprodurle. Il nuovo corpus contiene diecimila partite, pari a 380.000 decisioni non obbligate. Metà nasce da v13 contro se stessa; l'altra metà mescola quattro avversari con stili diversi.

Questa volta le partite intere vengono assegnate all'allenamento, alla verifica oppure all'esame finale. Nessuna mossa della stessa partita può attraversare quel confine. È un dettaglio importante: separare a caso le singole decisioni permetterebbe alla rete di studiare una parte di una partita e ritrovarne un'altra nell'esame.

Prima dell'allenamento v13 sceglie la stessa carta del consiglio nel 86,83% delle decisioni dell'esame. Dopo cinque passaggi sul corpus arriva al 92,88%. La dipendenza dai nomi dei semi scende dal 18,19% al 10,23%, sotto la soglia del 12% che i tentativi precedenti non erano riusciti a superare.

La prova sul gioco conserva il segnale. Su diecimila partite con gli stessi mazzi e i posti scambiati, la singola rete batte v13 di 0,51 punti a partita; il margine d'incertezza va da +0,11 a +0,92. Contro il consiglio a ventiquattro perde 0,23 punti, ma l'intervallo comprende la parità. In altre parole, con questa misura non riusciamo a distinguerli nettamente.

Anche il comportamento finisce nel punto giusto. L'overkill di briscola sui piatti poveri era all'8,0% per v13 e al 3,9% per il maestro; l'allievo distillato arriva al 5,5%. Ha assorbito una parte del miglioramento senza tornare al vecchio difetto.

Non è ancora una nuova versione ufficiale. Diecimila partite sono lo screening che autorizza il passo successivo: un corpus indipendente da cinquantamila partite. Se il risultato regge, il modello verrà finalmente provato dentro il PIMC usato dal sito. Fino ad allora resta un candidato locale.

<small>*Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/21-una-voce-sola.md)*</small>

L'estensione a cinquantamila partite ha poi rafforzato il risultato. L'accordo con il consiglio sale al 95,39% e la dipendenza dai nomi dei semi scende ancora, dal 10,23% al 6,04%. Sul tavolo batte v13 di 0,66 punti a partita, con un margine d'incertezza da +0,24 a +1,09; l'overkill sui piatti poveri arriva al 4,17%, ormai vicino al maestro.

Il primo controllo dentro il PIMC, su duemila partite, è neutro: +0,35 punti, ma con un intervallo che comprende ampiamente la parità. Resta quindi un ultimo test da diecimila partite nella configurazione esatta usata dal sito.

## Epilogo

Se c'è un filo comune, non è la serie dei campioni promossi. È il numero di idee respinte: penalità che peggioravano il comportamento, quaderni di esempi imparati a memoria, reti più grandi che non servivano, stime di carte avversarie utili in un punto e dannose in un altro, giudici di posizione troppo rumorosi a inizio partita. Ogni bocciatura ha tolto un'ipotesi dal tavolo e ha reso più chiaro il passo successivo.

Alla fine il progetto resta didattico proprio per questo. Non perché ogni scelta sia stata giusta, ma perché quasi ogni scelta è stata misurata: una variabile alla volta, stessi mazzi quando possibile, margini di incertezza dichiarati, e abbastanza pazienza da buttare via un'idea quando i numeri non la sostengono.

*Il diario continua, e il futuro non è un segreto: le prossime mosse sono
scritte, come tutto il resto, in un [piano operativo pubblico](https://github.com/ilCapo77/briscola.ai/blob/master/PLAN.md).
La progressione dei campioni — con i risultati misurati con cui ognuno si è guadagnato il posto — è in un
[report scaricabile (Excel)](https://github.com/ilCapo77/briscola.ai/raw/master/docs/reports/model_progress.xlsx); il
[codice e la storia completa](https://github.com/ilCapo77/briscola.ai) sono su GitHub.*
