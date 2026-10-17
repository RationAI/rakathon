# Jak využíváme umělou inteligenci

**MammoInvite** využívá AI ve třech vrstvách:

- **Predikce:** model strojového učení odhadne pro každou pozvanou ženu šanci, že do 3 a 12 měsíců přijde na screening (ROC AUC 0,84, kalibrované pravděpodobnosti).
- **Segmentace a plánování:** osm srozumitelných segmentů, každý s doporučeným zásahem. Manažer vybere měsíční dávku pozvánek a dostane akční seznam rozdělený do A/B skupin, takže účinek každého zásahu je měřitelný.
- **AI hlasoví asistenti:** zavolají ženám, na které dopis nezabírá, připomenou screening, poradí nejbližší pracoviště s nejkratší čekací dobou a rovnou je objednají přes objednávací a informační modul webové aplikace.

Při vývoji nám AI asistenti (Claude Code, Codex) pomáhali s analýzou 1,34 mil. pozvánek, modely i rešerší. Každé číslo ověřuje člověk na datech, která model neviděl.
