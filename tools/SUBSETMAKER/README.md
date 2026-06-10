# SubsetMaker  

> 🌐 Beste hizkuntzak: [Español](LEEME.md) | [English](README_EN.md)

YOLO Ikuskapen Artifizialeko datu-multzoen etiketa-orekatutako azpimultzoak sortzeko GUI aplikazioa.

![SubsetMaker pantaila-argazkia](https://github.com/user-attachments/assets/96bebab4-5fe4-416b-a2c5-462bf297fa62)

## Zer egiten du

SubsetMaker **YOLO datu-multzo kudeaketarako** mahaigaineko tresna bat da, sei funtzio integratuekin:

- **Azpimultzoak sortu** — datu-multzo bat nahi den irudien kopuru murriztu klase bakoitzeko, zein klase mantendu erabaki eta, aukeran, klase IDak birmapatu.
- **Datu-multzoa egiaztatu** — osotasun-arazoak hauteman eta konpondu, hala nola falta diren etiketa edo irudifitxategiak eta bateragarri ez diren irudiak dituzten etiketak.
- **YAML informazioa** — edozein `data.yaml` fitxategi aztertu klase-izenak eta konfigurazioa berrikusteko.
- **Datu-multzoa zatitu** — datu-multzo bat ausaz `train` / `val` azpimultzoeitan banatu, konfiguragarri den erlazio eta ausa-hazi erreproduziblea erabiliz.
- **Etiketak birzenbakitu** — klase IDak birmapatu direktorio bateko etiketa-fitxategi guztietan, bertan-bertan edo irteera-direktorio berri batera.
- **JSON → YAML** — COCO JSON anotazio-fitxategi bat (edo izen-zerrenda soil bat) YOLO-rekin bateragarria den `data.yaml` batera bihurtu.

Aplikazioak **gai ilunak eta argiak** onartzen ditu eta zure lehentasuna saioen artean gogoratzen du.

## Nola funtzionatzen du

### YOLO etiketa-formatua

YOLO datu-multzo bateko irudi bakoitzak izen-oinarri bera duen `.txt` lagun-fitxategi bat du. Fitxategi horretako lerro bakoitzak objektu bat deskribatzen du:

```
<class_id> <x_center> <y_center> <width> <height>
```

Koordenatu guztiak `[0, 1]` tartean normalizatuta daude, irudiaren dimentsioekiko erlatiboki. SubsetMaker-ek `<class_id>` eremua soilik irakurtzen eta berridazten du (lerro bakoitzeko lehen tokena); koordenatuak inoiz ez dira aldatzen.

### Azpimultzo sortu — algoritmoa

1. **Eskaneatze** — aplikazioak hautatutako zatiko irudi guztiak zeharkatzen ditu eta bateragarri den etiketa-fitxategian aurkitutako `class_id` bakoitza dituen irudi-bidean mapatzen du. Irudi bat mapa-n agertzen da dituen klase guztietarako.
2. **Lagin-hartu** — hautatutako klase bakoitzerako, klase hori duten irudien bilduma nahasten da (emandako ausazko hazia erabiliz) eta lehen `max_per_class` irudiak hartzen dira.
3. **Batasuna** — klase guztietako hautatutako irudiak multzo bakarrean biltzen dira, hainbat klase dituen irudi bat inoiz bikoiztu gabe.
4. **Etiketak iragazte** — kopiatutako irudi baten irteera-etiketa-fitxategia idaztean, mantendutako multzoan `class_id` duten anotazio-lerroak soilik idazten dira. Baztertutako klaseetako lerroak isilik ezabatzen dira.
5. **IDak birmapatu (aukerazkoa)** — *Klase IDak birmapatu* markatuta badago, mantendutako klase IDak ordenatzen dira eta `0`-tik aurrera zenbakitzen dira. Adibidez, `2`, `5` eta `7` jatorrizko klaseak mantentzen badituzu, `0`, `1` eta `2` bihurtzen dira irteeran. Mapaketa sortutako `data.yaml`-en islatzen da.
6. **Irteera idatzi** — irudiak `shutil.copy2`-rekin kopiatzen dira (metadatuak gordez), etiketa-fitxategiak iragazita/birmapatutako edukiarekin idazten dira, eta mantendutako klaseak soilik zerrendatzen dituen `data.yaml` berri bat sortzen da.

### Datu-multzoa egiaztatu — algoritmoa

Egiaztatzaileak irudi eta etiketa-direktorioak fitxategiz fitxategi alderatzen ditu:

- **Etiketa falta** — irudi-fitxategi bakoitzerako, etiketa-direktorioan izen-oinarri bera duen `.txt` fitxategi baten bila joaten da. Bat gabeko irudi oro falta bezala jakinarazten da.
- **Etiketa orfanoak** — `.txt` etiketa-fitxategi bakoitzerako, izen-oinarri bera duen irudi bat existitzen den egiaztatzen da (onartutako luzapen guztiak saiatuz). Bateragarri gabeko irudi gabeko etiketa bat orfano bezala jakinarazten da.

Zatiketa zehazten ez denean, egiaztapena **berriz ere** `images/` eta `labels/` zuhaitz osoan zehar egiten da, direktorio-egitura erlatiboa gordez. Zatiketa zehatza ematen denean, dagokion hosto-direktorioa soilik eskaneatzen da (ez-berriz ere).

Konponketa-ekintzak diseinuz seguruak dira: etiketa hutsak sortzeak benetan falta diren fitxategiak soilik ukitzen ditu, eta orfanoak ezabatzeak dagoeneko jakinarazitako fitxategiak soilik kentzen ditu.

### Datu-multzoa zatitu — algoritmoa

1. Iturriaren zatiko irudi-izen guztiak biltzen eta ordenatzen dira, gero `random.seed(seed)` erabiliz nahasten dira erreproduzgarritasunerako.
2. Lehen `round(total * train_pct / 100)` irudiak `train`-era joaten dira; gainerakoak `val`-era. Gutxienez irudi bat beti bermatuta dago zatiketa bakoitzean, totala ≥ 2 denean.
3. Irudi bakoitza irteera-karpetaren barruan `images/train` edo `images/val`-era kopiatzen da. Bateragarri den etiketa-fitxategia (badago) dagokion `labels/train` edo `labels/val`-era kopiatzen da. Etiketa gabeko irudiak errorerik gabe kopiatzen dira.
4. Jatorrizko datu-multzo-erroan `data.yaml` badago, irteera-karpetara ere kopiatzen da aldatu gabe.

### Etiketak birzenbakitu — algoritmoa

Birmapatze-tresnak hautatutako etiketa-direktorioko `.txt` fitxategi guztiak irakurtzen ditu. Anotazio-lerro bakoitzerako `class_id` erabiltzaileak emandako mapaketa-taulan bilatutako balioarekin ordezkatzen du. Taulan ez dauden klase IDak bere horretan uzten dira.

- **Bertan-bertan modua** (irteera-karpeta etiketa-karpetarekin bat datorrenean, edo hutsa denean): edukia benetan aldatzen diren fitxategiak soilik idazten dira, alferrikako diskoan idazketa saihestuz.
- **Kopiatu modua** (irteera-karpeta desberdina): etiketa-fitxategi guztiak IDak eguneratuta idazten dira helmugara, edukia aldatu den edo ez kontuan hartu gabe.

### YAML analizatzaile pertsonalizatua

SubsetMaker-ek YAML analizatzaile arin bat (`parse_yaml`) barne hartzen du PyYAML liburutegi osoan menpekotasunean egon beharrean. YOLO `data.yaml` fitxategietan erabiltzen diren YAML ezaugarriak onartzen ditu:

- `key: value` bikoteak (kateak eta zenbaki osoak)
- Fluxu-sekuentziak: `names: [cat, dog, bird]`
- Bloke-sekuentziak: `names:\n  - cat\n  - dog`
- Gako baten azpiko bloke-mapakuntzak: `names:\n  0: cat\n  1: dog`
- Lerro-barruko iruzkinak (`#`) eta komatxo arteko kateak (bakunak eta bikoitzak)

Ohiko YOLO konfigurazio-fitxategietan aurkitzen ez diren YAML ezaugarriak (aingurak, dokumentu anitzeko fitxategiak, habiaratze konplexua, etab.) ez dira onartzen.

## Onartutako datu-multzo-diseinua

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml        ← aukerazkoa (klase-izenentarako erabilia)
```

Diseinu lauak (irudiak eta etiketak zuzenean `images/` eta `labels/`-en azpian) ere onartzen dira.

**Onartutako irudi-formatuak:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

## Eskakizunak

- Python 3.10+
- `Pillow` ≥ 9.0 — irudi-fitxategien kudeaketarako
- `tkinter` — Python banaketa gehienekin sartuta dago (`python3-tk` instalatu Linuxen)

```bash
pip install -r requirements.txt
# Linuxerako soilik, tkinter falta bada:
sudo apt-get install python3-tk
```

## Erabilera

```bash
python subsetmaker.py
```

---

### ✂ Azpimultzo sortu

Zure datu-multzoaren iragazitako eta (aukeran) berroreka tuko azpimultzo bat kopiatzen du irteera-direktorio berri batera, birsortutako `data.yaml` barne.

**Lan-fluxua:**

1. **Datu-multzo-karpeta** — zure YOLO datu-multzoaren erroa hautatu.
2. **Irteera-karpeta** — aukeratu azpimultzoa idatziko den tokia.
3. **Zatiketa** — hautatu `train`, `val`, `test`, edo utzi hutsik diseinu lauentzat.
4. Sakatu **🔍 Kargatu datu-multzoa** — aplikazioak etiketak eskaneatzen ditu eta klase guztiak irudi-kopuruarekin zerrendatzen ditu.
5. **Klase-panela** — gorde nahi dituzun klaseak markatu/desmarka itzazu.
6. **Klase bakoitzeko irudi gehienez** — klase hautatuetarako sartu beharreko irudi-kopuruaren goiko muga ezarri.
7. **Ausazko hazia** — ezarri erreproduziblea den laginketa baterako osoko hazia.
8. **Klase IDak birmapatu** — markatuta dagoenean, irteera-etiketa-fitxategiek 0-tik aurrera birzenbakitutako klase IDak izango dituzte.
9. Sakatu **✂ Azpimultzo sortu** — irudiak eta iragazitako etiketak irteera-karpetara kopiatzen dira.

---

### 🔍 Datu-multzoa egiaztatu

Zatiketa bat osotasun-arazo ohikoengatik eskaneatzen du eta konponketa-aukera eskaintzen du.

**Lan-fluxua:**

1. **Datu-multzo-karpeta** — hautatu (edo berrerabili) zure YOLO datu-multzoaren erroa.
2. **Zatiketa** — hautatu egiaztatu beharreko zatiketa (`train`, `val`, `test`, edo hutsik diseinu lauentzat).
3. Sakatu **🔍 Egiaztatu datu-multzoa** — emaitzen panelak zerrendatzen ditu:
   - **Etiketa falta** — bateragarri den `.txt` etiketa-fitxategirik ez duten irudi-fitxategiak.
   - **Etiketa orfanoak** — bateragarri den irudirik ez duten `.txt` etiketa-fitxategiak.
4. Erabilatu konponketa-botoiak behar den arabera:
   - **➕ Sortu etiketa hutsak etiketatu gabeko irudientzat** — etiketa falta duen irudi bakoitzarentzat `.txt` huts bat idazten du (atzealde/lagin negatibo bezala markatuz).
   - **🗑 Ezabatu etiketa orfanoak** — bateragarri den irudirik ez duten etiketa-fitxategiak kentzen ditu.

---

### 📄 YAML informazioa

YOLO-estiloko `data.yaml` konfigurazio-fitxategi bat azkar aztertzen du.

**Lan-fluxua:**

1. Sakatu **…** `data.yaml` fitxategi batera nabigatzeko (edo idatzi bidea zuzenean).
2. Sakatu **📄 Kargatu YAML** — panelak erakusten du:
   - **nc** — fitxategian adierazitako klase-kopurua.
   - **names** — klase-izenen zerrenda osoa, lerro bakoitzeko bat, indizearekin.

---

### 🔀 Datu-multzoa zatitu

Datu-multzoaren zatiketa bat ausaz bereizitako `train` eta `val` azpimultzoeitan banatzen du.

**Lan-fluxua:**

1. **Datu-multzo-karpeta** — hautatu zure YOLO datu-multzoaren erroa.
2. **Irteera-karpeta** — aukeratu berri diren `train` / `val` azpi-direktorioak sortuko diren tokia.
3. **Zatiketa** — hautatu irakurriko den iturri-zatiketa (`train`, `val`, `test`, edo hutsik diseinu lauentzat).
4. **Train %** — ezarri entrenamendu-zatiketara joaten diren irudien ehunekoa (gainerakoak balioztatze-zatiketara joaten dira).
5. **Ausazko hazia** — ezarri erreproduziblea den nahasketarako osoko hazia.
6. Sakatu **🔀 Zatitu datu-multzoa** — irudiak eta haien etiketa-fitxategiak irteera-karpetaren barruan `images/train`, `images/val`, `labels/train` eta `labels/val`-era kopiatzen dira. Iturriaren `data.yaml` ere kopiatzen da badago.

---

### 🔢 Etiketak birzenbakitu

Pertsonalizatutako klase-ID birmapatze bat direktorio bateko YOLO etiketa-fitxategi guztiei aplikatzen die.

**Lan-fluxua:**

1. **Etiketa-karpeta** — `.txt` etiketa-fitxategiak dituen direktorioa hautatu.
2. **Irteera-karpeta** — aukeratu helmuga-karpeta, edo utzi etiketa-karpeta berera seinalatuz bertan-bertan birmapatzeko.
3. **Mapaketa** — sartu birmapaketa-arau bat lerro bakoitzeko `id_zaharra → id_berria` formatuan (adib. `2 → 0`).
4. Sakatu **🔢 Birzenbakitu etiketak** — aplikazioak edukia aldatzen duten fitxategiak soilik berridazten ditu (bertan-bertan modua) edo etiketa-fitxategi guztiak eguneratutako IDak dituela helmugara kopiatzen ditu.

---

### 📋 JSON → YAML

COCO JSON anotazio-fitxategi bat (edo JSON klase-izenen zerrenda soil bat) YOLO-rekin bateragarria den `data.yaml` batera bihurtzen du.

**Onartutako JSON formatuak:**

| Formatua | Adibidea |
|----------|----------|
| COCO anotazioak | `{"categories": [{"id": 1, "name": "cat"}, …]}` |
| Izenen zerrenda | `["cat", "dog", "bird"]` |
| Izenen objektua | `{"names": ["cat", "dog"]}` edo `{"names": {"0": "cat", "1": "dog"}}` |

**Lan-fluxua:**

1. Sakatu **…** **JSON fitxategia** ondoan zure JSON fitxategira nabigatzeko (edo idatzi bidea zuzenean).
2. Sakatu **📋 Kargatu JSON** — klase-mapaketa panelean erakusten da.
3. Aukeran editatu **Irteera YAML bidea**.
4. Sakatu **💾 Gorde YAML** — ateratako klase-izenekin `data.yaml` bat idazten da.

---

## Probak exekutatu

```bash
pip install pytest
pytest test_subsetmaker.py -v
```
