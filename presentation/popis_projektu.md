# MammoInvite

**Pozvánky na mamografii, které fungují. Každá žena jinak, každá včas.**

## Problém

VZP dnes posílá všem ženám stejný dopis. Do roka po pozvání přijde na screening jen asi 19 % z nich, u žen zvaných opakovaně jen několik procent. Jednotné zvaní nerozliší, komu stačí dopis, komu připomínka a kdo potřebuje osobní podporu.

## Data

- **1,34 milionu pozvánek** z let 2020–2024 pro 508 tisíc žen, s informací, zda žena přišla do 3, 6 a 12 měsíců.
- Věk, kraj a okres, historie screeningu a pozvánek, registrace a preventivní prohlídky u praktika a gynekologa, další screeningové programy, kontakty se zdravotnictvím.
- **75 mamografických center** (mamo.cz): poloha, ordinační doby a čekací doby. Z nich vzdálenost a dostupnost pro každý okres.
- Data ČSÚ o počtu žen a cizinkách v okresech.

## Jak to funguje

**1. Predikce: kdo nepřijde.** Model strojového učení odhadne pro každou pozvanou ženu šanci, že přijde. Porovnali jsme devět typů modelů (od rozhodovacího stromu po neuronovou síť nad historií pozvánek). Testujeme na ženách, které model při učení neviděl. Nejvíc rozhoduje doba od posledního vyšetření, počet předchozích pozvánek a preventivní návštěvy u gynekologa.

**2. Osm srozumitelných segmentů.** Jednoduchá pravidla rozdělí ženy do skupin s účastí od 2 % do 55 %. Například první pozvánka, pravidelné ženy, které vypadly z rytmu, ženy chodící ke gynekologovi, ale ne na mamograf, nebo ženy bez kontaktu se zdravotnictvím. Každý segment má navržený zásah a A/B test.

**3. Plánovač pozvánek pro pracovníky VZP.** Vybere měsíční rozesílku (asi 20 tisíc dopisů), ukáže rozložení šancí, segmenty a doporučený zásah pro každou ženu. Výstupem je akční seznam rozdělený do A/B skupin.

**4. Web a aplikace „Pár minut pro sebe".** Ženám vysvětlí, proč jít na screening i bez potíží, a odpoví na nejčastější obavy: „jsem zdravá", „bude to bolet", „bojím se výsledku". Objednání ve čtyřech krocích: kraj, centrum seřazené podle nejbližšího volného termínu, čas a SMS připomínka.

**5. AI hlasová asistentka.** Ženám, na které dopis nezabírá, zavolá, představí se jako AI, vysvětlí, čeho se nebát, a rovnou je objedná přes portál. Na požádání předá hovor člověku.

**6. Měření efektu.** Každý zásah běží jako A/B test v rámci segmentu. Po 12 měsících je rozdíl v účasti změřený, ne odhadnutý. Pilot u gynekologů potřebuje asi tři měsíční rozesílky.

## Dopad

- Scénář: cílené zásahy mohou přinést **zhruba 5 900 vyšetření ročně navíc** (o 12 % víc).
- **7 600 dopisů ročně** jde ženám vyšetřeným před méně než dvěma lety. Kontrola záznamů je ušetří.
- Mapy ukazují kraje a okresy, kde je účast nejnižší, takže podpora míří tam, kde chybí.
- Řešení stojí na datech, která VZP už má, a navazuje na dnešní měsíční rozesílání.
