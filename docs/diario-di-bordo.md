# Diario di bordo

Il diario di bordo — la storia del progetto (scelte, errori, svolte) raccontata in tono
divulgativo — è una **pagina del sito**: [ai.briscola.dev/diario](https://ai.briscola.dev/diario).

La **fonte unica è il markdown** `src/briscola_ai/frontend/static/diario.md`: per aggiornare
il diario si modifica solo quel file (il server lo renderizza a richiesta dentro
`diario_template.html`). Da aggiornare ai grandi traguardi, in stile non tecnico; la
cronologia tecnica completa vive nei messaggi di commit e in `docs/plans/`.
