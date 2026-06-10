# PCBTest

> 🌐 Beste hizkuntza batzuetan: [Castellano](README_ES.md) | [English](README_EN.md)

PCB txartelak bisualki ikuskatzeko eta YOLO dataseterako tresnak.

---

## Biltegiko egitura

```
PCBTest/
├── PCBTest/          # PCB txartelen ikuskatze aplikazio nagusia
└── tools/
    ├── 2dDatasetCreator/   # YOLO-rako 2D dataset sintetikoen sortzailea
    └── SUBSETMAKER/        # YOLO dataseten kudeaketarako GUIa
```

---

## PCBTest

Kamera bidez PCB txartelak bisualki ikuskatzeko aplikazioa.

**Pipeline:**
```
Kamera → homografia → orientazioa → YOLO → konparazioa → ONDO / TXARTO
```

Aplikazioak kamera baten bidez txartelaren irudia hartzen du, homografia erabiliz perspektiba zuzentzen du, YOLO modelo baten bidez osagaiak detektatzen ditu eta erreferentziako txartel batekin alderatzen ditu. Amaieran, txartela **ONDO** edo **TXARTO** dagoen jakinarazten du.

**Jetson Orin Nano** batean Docker bidez exekutatzeko diseinatuta.

Xehetasunak: [`PCBTest/GUIA_DE_USO.md`](PCBTest/GUIA_DE_USO.md).

---

## tools/2dDatasetCreator

YOLO modeloak entrenatzeko 2D irudi sintetikoen dataseterrak sortzen dituen scripta (`yodaut.py`).

`input/` karpetako osagai-irudiak hartzen ditu, parametro konfiguragarriekin (elementu kopurua, eskala, biraketa angelua) konbinatzen ditu eta entrenatzeko prest dagoen dataset bat sortzen du irudi eta YOLO etiketarekin.

Xehetasunak: [`tools/2dDatasetCreator/README.md`](tools/2dDatasetCreator/README.md).

---

## tools/SUBSETMAKER

YOLO dataseterrak kudeatzeko mahaigaineko aplikazioa (`subsetmaker.py`).

Funtzionalitate nagusiak:

- **Azpimultzo bat sortu** — dataset bat iragazi klasearen eta klase bakoitzeko irudi kopuru maximoaren arabera.
- **Dataseta egiaztatu** — etiketa umezurtzak edo etiketa gabeko irudiak detektatu.
- **Dataseta banatu** — split bat `train` / `val`-en banatu hazi erreproduzgarriarekin.
- **Etiketak birzenbakitu** — klase IDak biresleitu etiketa-fitxategi guztietan.
- **JSON → YAML** — COCO JSON anotazioak YOLOren `data.yaml` formatura bihurtu.
- **YAML Info** — edozein `data.yaml` fitxategi aztertu.

Xehetasunak: [`tools/SUBSETMAKER/README.md`](tools/SUBSETMAKER/README.md).

---

## Lizentzia

Iturburu kodea **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`) lizentziapean banatzen da.  
Dokumentazioa eta material azaltzaileak **Creative Commons Attribution-ShareAlike 4.0 International** (`CC-BY-SA-4.0`) lizentziapean banatzen dira.
