# Diario di bordo

*La storia vera di come un'IA ha imparato a giocare a Briscola — scelte, errori e svolte, raccontati senza tecnicismi.*

## Prima di tutto: la Briscola

La [Briscola](https://it.wikipedia.org/wiki/Briscola) è uno dei giochi di carte più amati d'Italia: quaranta carte, un seme "di briscola" che vince su tutti gli altri, e 120 punti in palio — chi ne fa più di 60 vince. Le regole si imparano in cinque minuti; giocarla bene è un'altra faccenda, ed è per questo che ci è sembrata perfetta per un esperimento di intelligenza artificiale.

Il suo fascino, per chi costruisce un'IA, sta in due ingredienti. Primo: l'**informazione nascosta** — non vedi la mano dell'avversario né l'ordine del mazzo, quindi non basta calcolare: bisogna *dedurre*, ricordare le carte uscite, leggere le abitudini di chi hai di fronte. Secondo: il **caso** — la pesca distribuisce fortuna e sfortuna, e distinguere una buona strategia da una buona mano richiede migliaia di partite e un po' di statistica. Scacchi e dama sono giochi di pura logica; la Briscola somiglia di più alla vita.

## Prologo — Una bottega, un apprendista <small>(gennaio 2026)</small>

Questo progetto è nato con un'ambizione semplice da dire e lunga da fare: costruire
*da zero*, pezzo per pezzo, un'intelligenza artificiale capace di giocare bene a Briscola — e capire davvero, strada facendo, come si insegna a giocare a una macchina. Niente scorciatoie: il motore delle regole, il tavolo da gioco online, la palestra di allenamento e l'allievo stesso sono tutti fatti in casa.

Il modo più onesto di leggere questa storia è immaginare una bottega artigiana: prima si costruisce il banco da lavoro, poi gli attrezzi, poi arriva un apprendista che impara — prima copiando, poi provando, infine facendo sparring con maestri sempre più forti. E come in ogni bottega vera, gli errori non si nascondono: si appendono al muro, perché sono lezioni.

C'è poi un dettaglio che rende questa storia doppiamente curiosa: in bottega non si è
lavorato da soli. Gran parte del codice e del ragionamento — le ipotesi, gli esperimenti, le
retromarce — è nata in dialogo con **agenti di intelligenza artificiale** (Claude, Codex,
Gemini), usati a più riprese come colleghi di banco. Un progetto che costruisce un'IA,
costruito insieme alle IA — e che è servito anche da *banco di prova* per loro: metterle
davanti a un progetto vero, con regole rigide, test severi e un maintainer esigente, dice
delle loro capacità molto più di qualunque demo.

Un'ultima avvertenza prima di cominciare: tutto quello che leggerai è ricostruito dalla
storia *scritta* del progetto — ogni modifica, ogni esperimento e ogni promozione di un
campione è registrata con data, numeri e motivazione. Dove la storia tace (e in un punto
tace davvero), lo diremo.

## Capitolo 1 — Un tavolo a cui sedersi <small>(gennaio 2026)</small>

Prima ancora di pensare all'intelligenza, serviva un tavolo. Un'IA che gioca a carte, da sola, è invisibile: produce numeri in un terminale, e nessun numero ti dice se *sembra* un giocatore vero. Per giudicarla — e per divertirsi, che non guasta — bisogna poterci sedere contro. Così una delle prime cose costruite fu proprio l'interfaccia: una pagina web con il tavolo verde, le carte in mano, la briscola scoperta e i punti che salgono. Niente da installare: si apre come un sito e si gioca.

Una precisazione importante, per non creare falsi ricordi: per quasi tutta questa storia quel "sito" **non era pubblico**. Girava solo sul computer di chi lo stava costruendo, come un prototipo in bottega — l'indirizzo che conosci oggi arriverà solo verso la fine del viaggio, quando il progetto sarà abbastanza maturo da reggere giocatori veri. All'inizio l'interfaccia serviva a una cosa sola: guardare l'apprendista negli occhi.

E fu proprio il tavolo a porre il primo grande dilemma, che non era di intelligenza ma di *teatro*. L'IA decide la sua carta in un lampo, qualche millesimo di secondo: se la giocasse davvero a quella velocità, la partita sarebbe illeggibile — carte che appaiono e spariscono prima che tu capisca cosa è successo. La pausa che vedi quando l'avversario "riflette" è una cortesia di regia, per darti il tempo di seguire. Ma *chi* deve dirigere quella regia? Per tre giorni il progetto provò a far comandare il ritmo alla pagina web: era lei a dire al server "ok, ora fai giocare l'IA". Sembrava naturale, ma si rivelò fragile — se la pagina si ricaricava al momento sbagliato o due segnali si accavallavano, la partita si ingarbugliava; e ogni pezza aggiungeva complicazione. Poi, la retromarcia: comanda il server, che gioca subito e fino in fondo senza aspettare nessuno; la pagina web riceve gli eventi e li *racconta* con i suoi tempi, come un telecronista che dosa la suspense su una partita già decisa. Quella scelta — il cervello avanza, la scena rallenta — regge ancora oggi, ed è stata la prima lezione del diario: a volte la soluzione elegante è tornare indietro.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/01-timing-ui.md)*</small>

## Capitolo 2 — Le regole come fonte di verità, e il patto anti-imbroglio

Poi venne il cuore. Può sembrare strano dedicare settimane alle *regole* di un gioco che sta su un foglietto, ma c'è una ragione: tutto quello che verrà dopo — allenamenti, esperimenti, verdetti — poggia su di esse, e un errore lì sotto avvelenerebbe ogni conclusione. Serviva quindi un motore delle regole puro, senza trucchi, e con una proprietà preziosa: data la stessa partita, rigiocarla produce *esattamente* le stesse carte e gli stessi esiti, mossa per mossa. Sembra pignoleria; è ciò che permette di rifare un esperimento, confrontare due giocatori sulla stessa identica mano, o indagare una partita sospetta come si riavvolge un nastro. E soprattutto venne firmato il patto che governa tutto il progetto: **l'IA non sbircia mai**. Ogni avversario artificiale riceve solo quello che vedrebbe un giocatore leale seduto al tavolo — le sue carte, il tavolo, la briscola, le carte già uscite. Mai il mazzo, mai la tua mano. La tentazione di barare, per chi costruisce un'IA, è
sottile e costante — non per malizia, ma per comodità: far vedere al programma "solo un
pezzettino" di informazione nascosta semplifica mille problemi. Per questo il patto non è
affidato alla buona volontà ma ai test: se una modifica lo viola, il semaforo diventa rosso. Tutta la forza che incontrerai giocando viene dall'allenamento, non da una scorciatoia. Questo patto è controllato da test automatici, ed è il motivo per cui di questo diario ci si può fidare.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/02-dominio-anticheat.md)*</small>

## Capitolo 3 — Le prime scuole: copiare e provare <small>(gennaio–febbraio 2026)</small>

Ma copiare da chi? Il primo maestro fu un *giocatore di regole*: un programma scritto a mano, pieno di istruzioni del tipo "se l'avversario ha giocato un carico e hai una briscola bassa, prendi" — la saggezza di base della Briscola, tradotta in codice. Non è intelligenza, è un regolamento interno; ma gioca in modo decoroso. L'apprendista imparò dapprima *copiando* migliaia di sue mosse. Funziona, e in fretta — ma chi copia eredita anche i difetti del maestro, e non lo supererà mai. Poi passò a
*provare*: milioni di partite contro avversari di ogni tipo, con un solo insegnamento — i punti fatti e subiti. È il metodo che, con mille raffinamenti, ha prodotto tutti i campioni successivi.

Di quell'epoca resta una battaglia memorabile: la guerra allo spreco delle briscole. L'allievo aveva il vizio di buttare briscole preziose per prese da due punti. Provammo a
*punirlo* durante l'allenamento, due volte, con due punizioni diverse: peggiorò entrambe le volte. Vinse invece un'idea più umile: un "guardiano" che al momento di giocare gli tocca la spalla — *sicuro di voler sprecare quel carico?* — e gli fa scegliere la briscola più economica che vince lo stesso. Costo in forza: zero. Lezione appesa al muro: non sempre si corregge un vizio rieducando; a volte basta un buon promemoria.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/03-prime-scuole-overkill.md)*</small>

## Capitolo 4 — Il silenzio <small>(febbraio–giugno 2026)</small>

Poi, per quattro mesi, il diario tace. Nessuna modifica, nessuna spiegazione. Le botteghe vere hanno anche questo: stagioni in cui si chiude a chiave e si va a vivere. Quando la porta si riaprì, a giugno, tutto cambiò passo.

## Capitolo 5 — L'estate della velocità <small>(giugno 2026)</small>

Qui serve una premessa: perché mai un'IA dovrebbe giocare *milioni* di partite? Perché impara dalla statistica, non dalle spiegazioni. Nessuno le dice "hai sbagliato a tagliare": vede solo i punti a fine partita, e in una singola partita la fortuna del mazzo pesa più della bravura. Per distinguere una strategia buona da una mano buona serve una quantità enorme di ripetizioni — è lo stesso motivo per cui un torneo si giudica su molte mani, non su una. La svolta di giugno fu quindi una palestra *quattordici volte più veloce*. Riscrivendo il cuore del gioco in una forma che il computer digerisce alla massima velocità, cinque milioni di partite di allenamento — che prima richiedevano ore — si giocano ora in un quarto d'ora. Sembra un dettaglio da ingegneri, ma è il motivo per cui tutto il resto della storia è potuto accadere: quando provare un'idea costa quindici minuti invece di una notte, si provano dieci idee al giorno. E nove si buttano senza rimpianti.

C'era però una condizione da rispettare, ed era sacra: la versione veloce doveva giocare
*esattamente* la stessa Briscola della versione lenta e leggibile. Ogni scorciatoia di
velocità è quindi incatenata all'originale da test di parità: stessa partita, mossa per
mossa, carta per carta, o non si passa. Velocità sì, ma mai al prezzo della verità.

Di quell'estate resta anche l'aneddoto più comico del diario: un campione che pesava 244 megabyte — mille volte più del dovuto — perché per errore si era salvato in pancia l'intero diario delle proprie metriche di allenamento. La copia ripulita pesava 138 kilobyte e giocava identica. Da allora, in bottega, si controlla la bilancia.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/04-fast-numba.md)*</small>

## Intermezzo — La porta si apre <small>(23–25 giugno 2026)</small>

A fine giugno, dopo quasi sei mesi di bottega a porte chiuse, arrivò il momento promesso nel primo capitolo: il tavolo diventò un sito vero, aperto a tutti. Sembra un dettaglio — "mettiamolo online" — ma tra un prototipo che gira sul computer di casa e un sito pubblico c'è un salto di responsabilità: in casa c'è un solo tavolo e un solo giocatore; online i server sono *tanti* e devono raccontarsi le partite a vicenda (se la tua partita vive su un server e la tua prossima mossa arriva a un altro, qualcuno deve tenere il filo), i modelli vanno distribuiti in modo verificabile, e ogni porta lasciata aperta per comodità di sviluppo va chiusa a chiave. Tre giorni di lavoro, e da allora l'indirizzo è quello che conosci: [ai.briscola.dev](https://ai.briscola.dev). Da quel momento la bottega ha una vetrina — e ogni campione promosso ha un pubblico.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/10-deploy-cloud.md)*</small>

## Capitolo 6 — La scala dei campioni, e la crisi <small>(fine giugno 2026)</small>

Con la palestra veloce nacque una dinastia: ogni campione si allenava contro il precedente e lo superava di poco. Terza, quarta, quinta, sesta generazione — progressi veri ma sempre più piccoli, a costo sempre più alto. E una sera, il dubbio più lucido dell'intero progetto, scritto nero su bianco: *stiamo migliorando davvero, o stiamo solo imparando a battere il nostro fratello maggiore?* A Briscola, come alla morra cinese, battere A non garantisce di battere B. Da quel dubbio nacquero misure più oneste: tornei incrociati, intervalli di confidenza, e una regola d'oro — *non avviare la prossima generazione solo per inerzia*.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/05-catena-campioni.md)*</small>

## Capitolo 7 — Il finale perfetto e la lezione del quaderno <small>(fine giugno 2026)</small>

Due strade nuove. La prima nasce da una proprietà curiosa della Briscola: nel finale, quando il mazzo è esaurito, il gioco smette di essere un gioco di informazione nascosta. Le carte sono quaranta, quelle uscite le hai viste, le tue le conosci: per esclusione, la mano dell'avversario si *deduce*. E quando non c'è più nulla di nascosto, la matematica può giocare *perfettamente* — esplorare ogni seguito possibile e scegliere il migliore, niente più intuito, solo calcolo. Il "solver" del finale vale quasi due punti a partita, gratis, e da allora ogni avversario del sito lo usa.

La seconda strada era più ambiziosa: far *pensare* l'IA anche prima del finale, simulando i possibili mondi nascosti — e poi travasare quel pensiero nell'istinto dell'allievo, fargli copiare le mosse pensate. Fallì, e fallì in un modo istruttivo: la rete imparò il quaderno di esempi *a memoria*, alla perfezione, senza capirne il senso — perfetta sugli esempi visti, mediocre su quelli nuovi. Come lo studente che recita la pagina ma non sa rispondere alla domanda girata. Lezione fondamentale, che tornerà:
**copiare mosse pensate non trasferisce il pensiero**.

Quello che invece funzionò fu usare il pensiero come *maestro di sparring*: la settima generazione si allenò contro un avversario che ragiona, e ne uscì col progresso più netto da mesi.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/06-search-endgame.md)*</small>

## Capitolo 8 — I due giorni di luglio <small>(1–3 luglio 2026)</small>

Poi arrivarono due giornate in cui il progetto mise alla prova praticamente ogni ipotesi rimasta sul tavolo — e lo fece con *esperimenti controllati*, nel senso che questa espressione ha nella scienza: per ogni idea si allena anche un "gemello" identico in tutto tranne che nell'idea da testare, i due giocano sulle stesse identiche mani, e la differenza — se c'è — si misura con un margine di incertezza dichiarato. Senza il gemello di controllo non sapresti mai se il merito è dell'idea o del caso. Il punto di partenza era una frustrazione sincera: i progressi erano diventati troppo timidi. La diagnosi individuò tre possibili colpevoli: all'allievo mancavano *informazioni* (non ricordava come l'avversario aveva giocato le prese passate), mancava *capienza* (un cervello troppo piccolo?), o mancava un *maestro* all'altezza.

I verdetti, uno per uno. La **memoria delle prese** — dare all'allievo occhi nuovi sul passato della partita — funziona: un guadagno piccolo ma inequivocabile, la prima vittoria del programma "più informazione" dopo molti tentativi a vuoto. Il **cervello più grande**, raddoppiato con una tecnica che preserva tutto l'istinto già appreso: quasi nulla — la capienza non era il collo di bottiglia. E il tentativo più curioso,
**sussurrare all'allievo le probabilità sulle carte avversarie** mentre gioca: addirittura dannoso, perché quel sussurro non conteneva nulla che l'allievo non potesse già dedurre da solo, e inseguirlo gli faceva perdere l'istinto. Infine il sogno del
**maestro che ragiona dall'inizio della partita** guidato da un "giudice di posizione": bocciato dalla fisica del gioco stesso — a inizio partita il futuro dipende da carte non ancora pescate, e nessun giudice può prevedere il caso. Il ragionamento paga solo quando l'incertezza si restringe.

Da tutti questi verdetti, un campione: l'**ottava generazione** — memoria delle prese, cervello raddoppiato, due giri di sparring — oggi il modello che affronti quando apri il sito. E una morale, diventata il motto di quei giorni:

> «Il vincolo non è l'allievo. È il maestro.»

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/07-belief-exit.md)*</small>

## Capitolo 9 — Il sussurro trova casa <small>(3 luglio 2026)</small>

Restava un'ultima carta: se il giudice di posizione fallisce dove regna il caso, la
*simulazione* no — giocare davvero i futuri possibili, tante volte, e fare la media. E qui il "sussurro" bocciato al capitolo precedente trovò finalmente il suo posto: non imboccare l'allievo, ma dire al simulatore *quali mondi vale la pena simulare* — quali carte, dato come ha giocato finora, l'avversario probabilmente ha in mano. Il risultato è l'avversario più forte mai offerto dal sito: per ogni mossa importante simula sessantaquattro possibili mani avversarie, pesate sul comportamento osservato, e in ciascuna gioca la partita fino in fondo prima di scegliere. Settantacinque millisecondi di calcolo per mossa — impercettibili per
te, un'eternità per un computer — in cui gioca internamente più partite di quante un essere
umano ne giochi in una vita. Lo trovi nel menu come
**"Modello locale + PIMC belief"**: batte il campione di casa di quasi quattro punti a partita. Nessuna idea di quei due giorni è andata sprecata: ognuna ha aperto una strada, o ne ha chiusa una per sempre.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/08-pimc-belief.md)*</small>

## Capitolo 10 — La nonna aveva ragione

Chiudiamo con la scoperta più affettuosa. Ci siamo chiesti: i precetti che ogni briscolista impara al bar — *non uscire coi carichi, non sprecare briscole per prese povere, non regalare figure* — andrebbero insegnati all'IA? Siamo andati a misurare. Verdetto: li rispetta già, tutti, e su alcuni è perfino più ortodossa del giocatore di regole scritto a mano. Nessuno glieli ha mai detti: li ha riscoperti da sola, in dieci milioni di partite, perché sono semplicemente *veri*. E quel 3–4% di volte in cui esce col carico violando il precetto? Probabilmente non è un errore: è il carico giocato *coperto*, sapendo di controllare le briscole. La saggezza popolare, più l'eccezione che la saggezza popolare non sa spiegare.

<small>🔬 *Per chi vuole i numeri e i dettagli: [approfondimento tecnico](https://github.com/ilCapo77/briscola.ai/blob/master/docs/diario/09-precetti-apertura.md)*</small>

## Epilogo — La disciplina dei "no"

Se questo diario ha una morale, non sta nei campioni promossi ma nel registro dei respinti: la storia del progetto conta molti più esperimenti bocciati che riusciti, e ognuno è stato bocciato *con i numeri*, non con le impressioni. Punizioni che peggioravano il vizio, quaderni imparati a memoria, cervelli più grandi che non servivano, sussurri che distraevano, giudici ciechi davanti al caso. Ogni "no" ha ristretto il campo finché le strade rimaste erano quelle giuste — ed è così che un progetto didattico su un gioco di carte è diventato, senza volerlo, una piccola lezione di metodo: *una variabile alla volta, misura tutto, e non affezionarti alle tue idee.*

*Il diario continua — e il futuro non è un segreto: le prossime mosse della bottega sono
scritte, come tutto il resto, in un [piano operativo pubblico](https://github.com/ilCapo77/briscola.ai/blob/master/PLAN.md).
La progressione dei campioni, con tutti i numeri di promozione, è in un
[report scaricabile (Excel)](https://github.com/ilCapo77/briscola.ai/raw/master/docs/reports/model_progress.xlsx); il
[codice e la storia completa](https://github.com/ilCapo77/briscola.ai) sono su GitHub.*
