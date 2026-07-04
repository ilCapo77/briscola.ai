# Diario di bordo

*La storia vera di come un'IA ha imparato a giocare a Briscola. Scelte, errori e svolte, raccontati senza tecnicismi.*

## Prima di tutto: la Briscola

La [Briscola](https://it.wikipedia.org/wiki/Briscola) è uno dei giochi di carte più amati d'Italia: quaranta carte, un seme "di briscola" che vince su tutti gli altri, 120 punti in palio. Chi ne fa più di 60 vince. Le regole si imparano in cinque minuti; giocarla bene è un'altra faccenda, ed è per questo che ci è sembrata perfetta per un esperimento di intelligenza artificiale.

Per chi costruisce un'IA il gioco ha due ingredienti interessanti. Il primo è l'**informazione nascosta**: non vedi la mano dell'avversario né l'ordine del mazzo, quindi non basta calcolare. Bisogna dedurre, ricordare le carte uscite, leggere le abitudini di chi hai di fronte. Il secondo è il **caso**: la pesca distribuisce fortuna e sfortuna, e per distinguere una buona strategia da una buona mano servono migliaia di partite e un po' di statistica. Scacchi e dama sono giochi di pura logica. La Briscola somiglia di più alla vita.

## Prologo <small>(gennaio 2026)</small>

Questo progetto è nato con un'ambizione semplice da dire e lunga da fare: costruire da zero un'intelligenza artificiale capace di giocare bene a Briscola, e capire davvero, strada facendo, come si insegna a giocare a una macchina. Niente scorciatoie: il motore delle regole, il tavolo da gioco online, la palestra di allenamento e l'allievo stesso sono tutti fatti in casa.

Il modo più onesto di leggere questa storia è immaginare una bottega artigiana. Prima si costruisce il banco da lavoro, poi gli attrezzi, poi arriva un apprendista che impara: prima copiando, poi provando, alla fine allenandosi con maestri sempre più forti. Gli errori, in questa bottega, non si nascondono. Ce ne sono parecchi, e alcuni sono stati più utili dei successi.

Un dettaglio che rende la storia doppiamente curiosa: in bottega non si è lavorato da soli. Gran parte del codice e del ragionamento è nata in dialogo con **agenti di intelligenza artificiale** (Claude, Codex, Gemini), usati a più riprese come colleghi di banco. Un progetto che costruisce un'IA, costruito insieme alle IA. Ed è servito anche da banco di prova per loro: metterli davanti a un progetto vero, con regole rigide, test severi e un maintainer esigente, dice delle loro capacità molto più di qualunque demo.

Un'ultima avvertenza prima di cominciare. Tutto quello che leggerai è ricostruito dalla storia *scritta* del progetto: ogni modifica, ogni esperimento e ogni promozione di un campione è registrata con data, numeri e motivazione. Dove la storia tace (e in un punto tace davvero), lo diremo.

## Capitolo 1 — Il tavolo da gioco <small>(gennaio 2026)</small>

Prima ancora di pensare all'intelligenza, serviva un tavolo. Un'IA che gioca a carte da sola è invisibile: produce numeri in un terminale, e nessun numero ti dice se *sembra* un giocatore vero. Per giudicarla bisogna poterci sedere contro. E poi, diciamolo, ci si voleva anche giocare. Così una delle prime cose costruite fu l'interfaccia: una pagina web con il tavolo verde, le carte in mano, la briscola scoperta e i punti che salgono.

Una precisazione, per non creare falsi ricordi: per quasi tutta questa storia quel "sito" non era pubblico. Girava solo sul computer di chi lo stava costruendo. L'indirizzo che conosci oggi arriva solo verso la fine del viaggio, quando il progetto era abbastanza maturo da reggere giocatori veri.

Fu proprio il tavolo a porre il primo dilemma serio, che non riguardava l'intelligenza ma la regia. L'IA decide la sua carta in qualche millesimo di secondo: se giocasse davvero a quella velocità la partita sarebbe illeggibile, con carte che appaiono e spariscono prima che tu capisca cosa è successo. La pausa che vedi quando l'avversario "riflette" è una cortesia, per darti il tempo di seguire. Ma chi deve gestire quella pausa? Per tre giorni il progetto provò a far comandare il ritmo alla pagina web: era lei a dire al server "ok, ora fai giocare l'IA". Sembrava naturale e si rivelò fragile. Bastava ricaricare la pagina al momento sbagliato perché la partita si ingarbugliasse, e ogni pezza aggiungeva complicazione. Alla fine, retromarcia: comanda il server, che gioca subito e fino in fondo; la pagina riceve gli eventi e li racconta con i suoi tempi, come un telecronista che dosa la suspense su una partita già decisa. Quella scelta regge ancora oggi.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/01-timing-ui.md)*</small>

## Capitolo 2 — Le regole e il patto anti-cheat

Poi venne il cuore. Può sembrare strano dedicare settimane alle regole di un gioco che sta su un foglietto, ma tutto quello che viene dopo (allenamenti, esperimenti, verdetti) poggia lì sopra, e un errore lì sotto avvelenerebbe ogni conclusione. Il motore delle regole ha anche una proprietà preziosa: data la stessa partita, rigiocarla produce esattamente le stesse carte e gli stessi esiti, mossa per mossa. Sembra pignoleria. In realtà è quello che permette di rifare un esperimento, confrontare due giocatori sulla stessa identica mano, o riavvolgere una partita sospetta come un nastro.

Soprattutto, in quei giorni venne firmato il patto che governa tutto il progetto, e che nel codice e nelle note del sito ha un nome preciso: **anti-cheat**. In parole povere: l'IA non sbircia mai. Ogni avversario artificiale riceve solo quello che vedrebbe un giocatore leale seduto al tavolo — le sue carte, il tavolo, la briscola, le carte già uscite. Mai il mazzo, mai la tua mano. La tentazione di barare, per chi costruisce un'IA, è sottile e costante; non per malizia ma per comodità, perché far vedere al programma "solo un pezzettino" di informazione nascosta semplifica mille problemi. Per questo il patto non è affidato alla buona volontà: è controllato da test automatici, e se una modifica lo viola il semaforo diventa rosso. Tutta la forza che incontrerai giocando viene dall'allenamento, non da una scorciatoia.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/02-dominio-anticheat.md)*</small>

## Capitolo 3 — Copiare, provare, fare sparring <small>(gennaio–febbraio 2026)</small>

Il primo maestro fu un *giocatore di regole*: un programma scritto a mano, pieno di istruzioni tipo "se l'avversario ha giocato un carico e hai una briscola bassa, prendi". La saggezza di base della Briscola tradotta in codice. Non è intelligenza, è un regolamento interno, però gioca in modo decoroso. L'apprendista cominciò *copiando* migliaia di sue mosse. Funziona, e anche in fretta, ma chi copia eredita pure i difetti del maestro, e soprattutto non lo supererà mai.

Poi si passò al metodo che avrebbe prodotto tutti i campioni successivi, e che nel diario chiamiamo **sparring**, come nella boxe. Lo sparring partner è il compagno con cui ti alleni combattendo davvero: non ti spiega la teoria, sale sul ring e ti mette alla prova. Qui funziona uguale: l'allievo gioca milioni di partite vere contro un avversario, e l'unico insegnamento che riceve è il punteggio. Nessuno gli dice mai "qui dovevi tagliare". Se spreca una briscola perde punti, e a forza di perderne capisce da solo. Due dettagli contano molto. Primo: lo sparring partner è *congelato*, durante l'allenamento non cambia e non impara, migliora solo l'allievo (se cambiassero entrambi sarebbe come allenarsi su un ring che si muove). Secondo: la qualità del partner conta. Contro uno scarso impari poco perché vinci sempre; contro uno troppo più forte nemmeno, perché perdi sempre e non capisci il perché. I salti migliori sono arrivati con partner un gradino sopra l'allievo.

Di quell'epoca resta una battaglia memorabile: la guerra allo spreco delle briscole. L'allievo aveva il vizio di buttare briscole preziose per prese da due punti. Provammo a punirlo durante l'allenamento, due volte, con due punizioni diverse. Peggiorò entrambe le volte. Vinse un'idea più umile: un "guardiano" che al momento di giocare gli tocca la spalla — sicuro di voler sprecare quel carico? — e gli fa scegliere la briscola più economica che vince lo stesso. Costo in forza: zero. A volte non serve rieducare, basta un buon promemoria.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/03-prime-scuole-overkill.md)*</small>

## Capitolo 4 — Quattro mesi di pausa <small>(febbraio–giugno 2026)</small>

Poi, per quattro mesi, il diario tace. Nessuna modifica, nessuna spiegazione. Capita anche alle botteghe vere. Quando la porta si riaprì, a giugno, tutto cambiò passo.

## Capitolo 5 — La palestra veloce <small>(giugno 2026)</small>

Prima una domanda legittima: perché mai un'IA dovrebbe giocare *milioni* di partite? Perché impara dalla statistica, non dalle spiegazioni. Vede solo i punti a fine partita, e in una singola partita la fortuna del mazzo pesa più della bravura. Per distinguere una strategia buona da una mano buona servono quantità enormi di ripetizioni, per lo stesso motivo per cui un torneo non si giudica su una mano sola.

La svolta di giugno fu una palestra quattordici volte più veloce. Riscrivendo il cuore del gioco in una forma che il computer digerisce alla massima velocità, cinque milioni di partite di allenamento passarono da ore a un quarto d'ora. Sembra un dettaglio da ingegneri ed è invece il motivo per cui tutto il resto della storia è potuto accadere: quando provare un'idea costa quindici minuti invece di una notte, ne provi dieci al giorno, e nove le butti senza rimpianti.

C'era una condizione da rispettare: la versione veloce doveva giocare esattamente la stessa Briscola della versione lenta e leggibile. Ogni scorciatoia è quindi incatenata all'originale da test di parità — stessa partita, mossa per mossa, carta per carta, o non si passa.

Di quell'estate resta anche l'aneddoto più comico del diario: un campione che pesava 244 megabyte, mille volte più del dovuto, perché per errore si era salvato in pancia l'intero registro delle proprie metriche di allenamento. La copia ripulita pesava 138 kilobyte e giocava identica. Da allora in bottega si controlla la bilancia.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/04-fast-numba.md)*</small>

## Intermezzo — Il sito va online <small>(23–25 giugno 2026)</small>

A fine giugno arrivò il momento promesso nel primo capitolo: il tavolo diventò un sito vero, aperto a tutti. Sembra un dettaglio, "mettiamolo online", ma tra un prototipo sul computer di casa e un sito pubblico c'è un salto di responsabilità. In casa c'è un solo tavolo e un solo giocatore. Online i server sono tanti e devono raccontarsi le partite a vicenda (se la tua partita vive su un server e la tua prossima mossa arriva a un altro, qualcuno deve tenere il filo), i modelli vanno distribuiti in modo verificabile, e le porte lasciate aperte per comodità di sviluppo vanno chiuse a chiave. Tre giorni di lavoro. Da allora l'indirizzo è quello che conosci: [ai.briscola.dev](https://ai.briscola.dev).

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/10-deploy-cloud.md)*</small>

## Capitolo 6 — Un campione dopo l'altro, e un dubbio <small>(fine giugno 2026)</small>

Con la palestra veloce i campioni cominciarono a susseguirsi: ognuno si allenava contro il precedente e lo superava di poco. Terza, quarta, quinta, sesta generazione. Progressi veri ma sempre più piccoli, a costo sempre più alto. Finché una sera qualcuno scrisse nero su bianco il dubbio più lucido dell'intero progetto: *stiamo migliorando davvero, o stiamo solo imparando a battere il nostro fratello maggiore?* A Briscola, come alla morra cinese, battere A non garantisce di battere B. Da quel dubbio nacquero misure più oneste (tornei incrociati, margini di incertezza dichiarati) e una regola che da allora nessuno ha più infranto: non avviare la prossima generazione solo per inerzia.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/05-catena-campioni.md)*</small>

## Capitolo 7 — Il solver del finale, e perché copiare non basta <small>(fine giugno 2026)</small>

Due strade nuove. La prima nasce da una proprietà curiosa della Briscola: nel finale, quando il mazzo è esaurito, l'informazione nascosta sparisce. Le carte sono quaranta, quelle uscite le hai viste, le tue le conosci: la mano dell'avversario si deduce per esclusione. A quel punto la Briscola smette di essere un gioco di intuito e diventa, per qualche presa, un problema di calcolo puro.

Ed è qui che entra il **solver** (dall'inglese *to solve*, risolvere). Non è un giocatore che "gioca bene": è un calcolatore che *risolve* la posizione, come si risolve un'equazione. Funziona così: prende le carte rimaste e prova mentalmente tutte le strade — se gioco questa, lui può rispondere con quella o quell'altra, e allora io... — fino all'ultima carta di ogni ramificazione, assumendo che l'avversario giochi sempre la sua difesa migliore. Con tre carte per parte le combinazioni sono poche migliaia, roba da millesimi di secondo per un computer. Alla fine non sceglie una mossa "promettente": sceglie la mossa *dimostrata* migliore. Contro un solver nel finale non esiste giocata furba che tenga, per definizione — al massimo puoi pareggiare la perfezione, mai superarla.

Perché allora non usarlo per tutta la partita? Per due muri. Il primo è l'informazione nascosta: finché il mazzo non è esaurito non sai cosa ha in mano l'avversario né cosa pescherai, e non puoi "provare tutte le strade" di un labirinto di cui non conosci i corridoi. Il secondo è la dimensione: a inizio partita le ramificazioni possibili sono un numero astronomico. Il finale è l'unico momento in cui il labirinto è insieme piccolo e completamente illuminato.

Misurato sul campo, il solver del finale vale quasi due punti a partita, gratis. Da allora ogni avversario del sito lo usa: qualunque cosa accada prima, le ultime prese le gioca la matematica.

La seconda strada era più ambiziosa: far *pensare* l'IA anche prima del finale, simulando i possibili mondi nascosti, e poi travasare quel pensiero nell'istinto dell'allievo facendogli copiare le mosse pensate. Fallì in un modo istruttivo. La rete imparò il quaderno di esempi a memoria, alla perfezione, senza capirne il senso: perfetta sugli esempi visti, mediocre su quelli nuovi. Come lo studente che recita la pagina ma non sa rispondere alla domanda girata. La lezione, che tornerà più volte: copiare mosse pensate non trasferisce il pensiero.

Quello che funzionò fu usare il pensiero come sparring partner: la settima generazione si allenò contro un avversario che ragiona, e ne uscì col progresso più netto da mesi.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/06-search-endgame.md)*</small>

## Capitolo 8 — Tutte le ipotesi alla prova <small>(1–3 luglio 2026)</small>

Poi arrivarono due giornate in cui il progetto mise alla prova quasi ogni ipotesi rimasta sul tavolo. Il punto di partenza era una frustrazione sincera: i progressi erano diventati troppo timidi, e la diagnosi indicava tre possibili colpevoli. All'allievo mancavano informazioni (non ricordava come l'avversario aveva giocato le prese passate)? Mancava capienza (un cervello troppo piccolo)? O mancava un maestro all'altezza?

Una parola su come si decide, in casi così. Ogni idea è stata provata con un *esperimento controllato*, nel senso scientifico: si allena anche un "gemello" identico in tutto tranne che nell'idea da testare, i due giocano sulle stesse identiche mani, e la differenza si misura con un margine di incertezza dichiarato. Senza il gemello non sapresti mai se il merito è dell'idea o del caso.

I verdetti. La **memoria delle prese** funziona: un guadagno piccolo ma inequivocabile, la prima vittoria del filone "più informazione" dopo molti buchi nell'acqua. Il **cervello più grande** (raddoppiato con una tecnica che preserva l'istinto già appreso): quasi niente, la capienza non era il problema. Il tentativo più curioso, **sussurrare all'allievo le probabilità sulle carte avversarie** mentre gioca, si rivelò addirittura dannoso: quel sussurro non conteneva nulla che l'allievo non potesse dedurre da solo, e inseguirlo gli faceva perdere l'istinto. Infine il sogno del **maestro che ragiona dall'inizio della partita**, guidato da un "giudice di posizione": bocciato dalla natura stessa del gioco, perché a inizio partita il futuro dipende da carte non ancora pescate e nessun giudice può prevedere il caso.

Da tutto questo uscì comunque un campione, l'**ottava generazione** — memoria delle prese, cervello raddoppiato, due giri di sparring — che per un giorno fu il modello del sito (il capitolo 11 racconta chi l'ha detronizzata). E uscì il motto di quei giorni:

> «Il vincolo non è l'allievo. È il maestro.»

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/07-belief-exit.md)*</small>

## Capitolo 9 — L'avversario che simula i futuri <small>(3 luglio 2026)</small>

Restava un'ultima carta. Se il giudice di posizione fallisce dove regna il caso, la *simulazione* no: giocare davvero i futuri possibili, tante volte, e fare la media. E qui il sussurro bocciato al capitolo precedente trovò il suo posto. Non imboccare l'allievo, ma dire al simulatore quali mondi vale la pena simulare: quali carte, visto come ha giocato finora, l'avversario probabilmente ha in mano.

Il risultato è l'avversario più forte mai offerto dal sito. Per ogni mossa importante immagina sessantaquattro possibili mani avversarie, pesate sul comportamento osservato, e in ciascuna gioca la partita fino in fondo prima di scegliere. Sono una manciata di millisecondi per te, un'eternità di partite immaginarie per lui. Lo trovi nel menu come **"Modello locale + PIMC belief"**, e batte il campione di casa di quasi quattro punti a partita. Col senno di poi, nessuna idea di quei due giorni è andata sprecata: ognuna ha aperto una strada o ne ha chiusa una per sempre.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/08-pimc-belief.md)*</small>

## Capitolo 10 — La nonna aveva ragione

Una scoperta affettuosa, prima del gran finale. Ci siamo chiesti: i precetti che ogni briscolista impara al bar — non uscire coi carichi, non sprecare briscole per prese povere, non regalare figure — andrebbero insegnati all'IA? Siamo andati a misurare, ed è venuto fuori che li rispetta già tutti. Su un paio è perfino più ortodossa del giocatore di regole scritto a mano. Nessuno glieli ha mai detti: li ha riscoperti da sola, giocando, perché sono semplicemente veri. E quel 3–4% di volte in cui esce col carico, violando il precetto? Probabilmente non è indisciplina: è il carico giocato coperto, sapendo di controllare le briscole. La saggezza popolare, più l'eccezione che la saggezza popolare non sa spiegare.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/09-precetti-apertura.md)*</small>

## Capitolo 11 — I due allievi <small>(3–4 luglio 2026)</small>

Chiusi i conti con le grandi teorie, restava una domanda antica: si impara di più da un maestro straordinario, o giocando tantissimo con un po' di tutti? Invece di discuterne, la bottega ha fatto l'esperimento. Stesso allievo di partenza, due percorsi. Il primo — ricetta proposta dal padrone di casa, va detto — venti milioni di partite contro un cartellone variegato: lo sparring forte, se stesso allo specchio, e un buon 20% di ritorno al bar, contro i vecchi giocatori di regole, per non dimenticare come si vince contro chi gioca semplice. Il secondo: un quarto delle partite, tutte contro il maestro più forte mai costruito, quello che simula i futuri prima di ogni mossa.

Trenta ore di computer dopo, il verdetto era netto e un po' umiliante per il maestro d'élite: l'allievo del bar ha vinto tutto. Contro il campione in carica, contro il record storico, e nello scontro diretto col suo gemello. Il gemello del gran maestro aveva sì imparato più in fretta per partita — segno che un buon insegnante conta — ma la sua dieta raffinata e monotona gli aveva fatto perfino disimparare come si stracciano i principianti. Il vincitore è oggi il campione in carica: la **nona generazione**, la prima a migliorare su tutti i metri contemporaneamente.

La morale stavolta l'ha scritta l'esperimento, non il diarista: la saggezza non abita solo nei maestri illustri. Abita anche nel chilometraggio, nella varietà, e in quel 20% di partite al bar che nessun pedagogo raffinato avrebbe prescritto.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/11-due-allievi.md)*</small>

## Epilogo

Se questo diario ha una morale, non sta nei campioni promossi ma nel registro dei respinti. La storia del progetto conta più esperimenti bocciati che riusciti, e ognuno è stato bocciato con i numeri, non con le impressioni: punizioni che peggioravano il vizio, quaderni imparati a memoria, cervelli più grandi che non servivano, sussurri che distraevano, giudici ciechi davanti al caso. Ogni "no" ha ristretto il campo, finché le strade rimaste erano quelle giuste. Così un progetto didattico su un gioco di carte è diventato, senza volerlo, una piccola lezione di metodo: una variabile alla volta, misura tutto, e non affezionarti alle tue idee.

*Il diario continua, e il futuro non è un segreto: le prossime mosse della bottega sono
scritte, come tutto il resto, in un [piano operativo pubblico](https://github.com/ilCapo77/briscola.ai/blob/master/PLAN.md).
La progressione dei campioni, con tutti i numeri di promozione, è in un
[report scaricabile (Excel)](https://github.com/ilCapo77/briscola.ai/raw/master/docs/reports/model_progress.xlsx); il
[codice e la storia completa](https://github.com/ilCapo77/briscola.ai) sono su GitHub.*
