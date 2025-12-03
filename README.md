# Perceptron Tanító Alkalmazás

Interaktív oktatási alkalmazás a perceptron neurális hálózat algoritmus megértéséhez és tanításához. Az alkalmazás lehetővé teszi a perceptron működésének lépésről lépésre való követését, a döntési határ vizualizálását és a tanulási folyamat megfigyelését.

## Bevezetés

A perceptron a legegyszerűbb neurális hálózat modell, amelyet Frank Rosenblatt fejlesztett ki 1957-ben. Ez az alkalmazás interaktív módon mutatja be az algoritmus működését, lehetővé téve a felhasználók számára, hogy megértsék a gépi tanulás alapjait.

## Gyors Indítás

### Előfeltételek

- Python 3.8 vagy újabb verzió
- pip csomagkezelő

### Telepítés

```bash
# Függőségek telepítése
pip install -r requirements.txt

# Webes alkalmazás indítása
streamlit run scripts/app.py

# Vagy konzolos alkalmazás
python scripts/main.py
```

Windows-on használhatja a `start.bat` fájlt is a webes alkalmazás indításához.

## Funkciók

- **Perceptron algoritmus**: Teljes implementáció Python nyelven, súlyok és bias kezeléssel
- **Webes felület**: Streamlit alapú interaktív felület, valós idejű frissítéssel
- **Konzolos alkalmazás**: Menüvezérelt interfész részletes információk megjelenítésével
- **Vizualizációk**: Adatpontok megjelenítése, döntési határ ábrázolása, konvergencia görbe
- **Epoch-onkénti animáció**: A döntési határ változásának követése lépésről lépésre
- **Többféle adathalmaz**: Könnyű, nehéz, függőleges, átlós elválasztású adatok
- **Magyar nyelvű**: Teljes magyar nyelvű interfész és dokumentáció

## Projekt Struktúra

```
Percetron_App/
├── data/                    # CSV adatfájlok
│   ├── data.csv
│   ├── data_easy.csv
│   ├── data_hard.csv
│   ├── data_vertical.csv
│   └── data_diagonal.csv
├── scripts/                 # Fő alkalmazások
│   ├── app.py              # Streamlit webes alkalmazás
│   ├── main.py             # Konzolos alkalmazás
│   └── data_generator.py   # Adatgeneráló script
├── src/                     # Forráskód modulok
│   ├── perceptron.py       # Perceptron osztály implementációja
│   ├── data_loader.py      # Adatbetöltő osztály
│   └── visualizer.py       # Vizualizációs osztály
├── docs/                    # Dokumentáció
├── requirements.txt         # Python függőségek
├── start.bat               # Windows indító script
├── start.sh                # Linux/Mac indító script
└── README.md               # Ez a fájl
```

## Technológiai Stack

- **Python 3.8+**: Programozási nyelv
- **NumPy**: Numerikus számítások, mátrix műveletek, véletlenszám generálás
- **Pandas**: CSV fájlok betöltése, adatstruktúrák kezelése
- **Matplotlib**: Statikus grafikonok generálása (konzolos alkalmazás)
- **Streamlit**: Webes felület fejlesztése, interaktív widgetek
- **Plotly**: Interaktív grafikonok (webes alkalmazás)

## Használat

### Webes Felület

1. **Adatok betöltése**: Az oldalsávban válasszon egy CSV fájlt a legördülő menüből, majd kattintson az "Adatok betöltése" gombra
2. **Perceptron inicializálása**: Állítsa be a tanulási rátát (ajánlott: 0.01) és a maximális epoch számot (ajánlott: 100), majd kattintson a "Perceptron inicializálása" gombra
3. **Tanítás**: A "Tanítás" fülön választhat "Egy lépés tanítás" (lépésenkénti) vagy "Teljes tanítás" (automatikus) opciót
4. **Eredmények megtekintése**: 
   - "Vizualizáció" fül: Döntési határ megjelenítése az adatpontokkal, epoch-onkénti animáció
   - "Konvergencia" fül: Konvergencia görbe, amely mutatja a hibák számának változását

### Konzolos Alkalmazás

A menüvezérelt interfész lehetővé teszi:
- Adatfájlok betöltését
- Lépésenkénti vagy teljes tanítást
- Interaktív tanítási módot (epochonkénti megállás)
- Grafikonok megjelenítését

## Saját Adatok Használata

Hozzon létre egy CSV fájlt a `data/` mappában a következő formátumban:

```csv
x1,x2,label
2.5,3.2,1
1.8,2.1,1
0.5,0.3,0
0.2,0.1,0
```

A fájl első sora tartalmazza az oszlopneveket, a további sorok az adatértékeket. A label oszlop csak 0 vagy 1 értékeket tartalmazhat.

## Dokumentáció

Részletes dokumentáció a `docs/` mappában található:
- Fő dokumentáció
- Használati útmutató
- Tesztelési dokumentáció
- Előadás anyag
- Szakdolgozat sablon

## Perceptron Algoritmus

A perceptron egy lineáris bináris osztályozó, amely:
- Kezdetben véletlenszerű súlyokat és bias értéket használ
- Minden adatpontra előrejelzést készít
- Hibás predikció esetén módosítja a súlyokat és bias-t
- Epoch-onként ismétli a folyamatot, amíg konvergál (0 hiba)

A konvergencia tétel szerint, ha az adatok lineárisan szeparálhatók, a perceptron véges számú lépésben konvergál.
