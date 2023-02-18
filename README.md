# Anleitung

## Tokenizer
Damit das Skript läuft sollten die folgenden Anforderungen erfüllt werden:
- Input CSV sollte mindestens die Spalte "content" enthalten
- Path Variablen anpassen
- Vorher pandas per pip3 installieren: pip3 install pandas
- Vorher sPACY model importieren: python3 -m spacy download en_core_web_sm

Für Tokenization einfach die Methoden aufrufen. Entsprechende txt files mit tokenized Inhalten werden am konfigurierten
Pfad erstellt.

Für Erstellung einzelner txt files:
1) Erstelle csv mit gleich entweder positiven oder negativen Ergebnissen
2) Konfiguriere csv-Pfad, dessen Inhalt gelesen werden soll
3) Konfiguriere folder-Pfad (am besten Ordner erstellen ("pos" oder "neg") und diesen Pfad einfügen)