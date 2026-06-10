# pcbTest – Erabiltzaile gida

> 🌐 Beste hizkuntza batzuetan: [Castellano](GUIA_DE_USO.md) | [English](USER_GUIDE.md)

*GPL-3.0-or-later / CC-BY-SA-4.0*

---

## Helburua

PCB plaken ikuskapen bisuala egitea kameraren bidez, homografia erabiliz irudia zuzentzeko, YOLO bidez osagaiak detektatzeko eta erreferentziazko plaka batekin alderatzeko.

**Pipelinea:**
```
Kamera → homografia → orientazioa → YOLO → konparazioa → OK / MAL
```

**Lizentzia:** Kodea: GPL-3.0-or-later. Dokumentazioa eta materialak: CC-BY-SA-4.0.

---

## Aurkibidea

1. [Zer egiten du pcbTest programak](#1-zer-egiten-du-pcbtest-programak)
2. [Proiektuaren egitura](#2-proiektuaren-egitura)
3. [Lehen abioa](#3-lehen-abioa)
4. [Rutas fitxa](#4-rutas-fitxa)
5. [Kamera fitxa](#5-kamera-fitxa)
6. [Ikuskapenaren konfigurazioa](#6-ikuskapenaren-konfigurazioa)
7. [Ikuskapena egitea](#7-ikuskapena-egitea)
8. [Emaitzak interpretatzea](#8-emaitzak-interpretatzea)
9. [Doikuntza gomendatuak](#9-doikuntza-gomendatuak)
10. [Arazo ohikoak](#10-arazo-ohikoak)
11. [Lizentzia eta erabilera-oharrak](#11-lizentzia-eta-erabilera-oharrak)

---

## 1. Zer egiten du pcbTest programak

pcbTest PCB baten egoera aztertzeko tresna da. Kamera batek plaka baten irudia hartzen du, programak perspektiba zuzentzen du, osagaiak detektatzen ditu eta erreferentziazko plaka zuzen batekin alderatzen du. Azkenean, erabiltzaileari plaka **OK** edo **MAL** dagoen erakusten dio.

Programak ez du soilik YOLO inferentzia egiten. Aurretik irudia prestatzen du, plaka plano berean jartzen du eta orientazioa zuzentzen saiatzen da. Horregatik, pipeline osoa ulertzea garrantzitsua da.

```
Kamera
  → irudia hartu
  → plaka detektatu
  → homografia aplikatu
  → orientazioa egiaztatu serigrafiaren bidez
  → YOLO bidez osagaiak detektatu
  → referenceBoard/ karpetako erreferentziarekin alderatu
  → emaitza: PLACA OK edo PLACA MAL
```

> **Garrantzitsua:** pcbTest prototipatze eta hezkuntza erabilerarako pentsatuta dago. Industria-ingurune batean erabili aurretik, argiztapena, kamera, modeloa, tolerantziak eta faltsu positibo/negatiboak behar bezala balioztatu behar dira.

---

## 2. Proiektuaren egitura

Beste Jetson batean programa kopiatzeko, karpeta garbi bat izatea komeni da:

```
pcbTest/
├── pcb_gui_inspeccion.py
├── pcb_gui_inspeccion.sh
├── pcb_realtime_pipeline.py
├── pcb_realtime_pipeline.sh
├── pcb_camera_test.py
├── pcb_camera_test.sh
├── procesar_pcb_homografia_yolo.py
├── comparar_yolo_reference.py
├── config_homografia.json
├── keypoints/
│   └── serigrafia.png
├── referenceBoard/
│   ├── notes.json
│   └── labels/
│       └── referencia.txt
├── weights/
│   └── best.pt
├── results/
│   └── .gitkeep
├── README.md
├── install_notes.md
└── .gitignore
```

**Fitxategi garrantzitsuenak:**

| Fitxategia | Deskribapena |
|------------|--------------|
| `pcb_gui_inspeccion.py` | Interfaze grafiko nagusia. |
| `pcb_realtime_pipeline.py` | Ikuskapen-prozesu osoa egiten duen pipelinea. |
| `pcb_camera_test.py` | Kamera azkar probatzeko script-a. |
| `procesar_pcb_homografia_yolo.py` | Homografia, orientazioa eta irudi prozesamendua. |
| `comparar_yolo_reference.py` | YOLO detekzioak erreferentziarekin alderatzen ditu. |
| `config_homografia.json` | Irudi zuzenduko tamaina definitzen du. |
| `referenceBoard/` | Plaka zuzenaren erreferentzia geometrikoa eta klase-izenak. |
| `weights/best.pt` | YOLO eredua. |

---

## 3. Lehen abioa

Programa abiarazi aurretik, ziurtatu Jetsonak Docker erabiltzeko baimena duela eta kamera sisteman agertzen dela.

**Baimenak:**

```bash
cd pcbTest
chmod +x pcb_gui_inspeccion.py
chmod +x pcb_gui_inspeccion.sh
chmod +x pcb_realtime_pipeline.py
chmod +x pcb_realtime_pipeline.sh
chmod +x pcb_camera_test.py
chmod +x pcb_camera_test.sh
```

**GUI abiaraztea:**

```bash
cd pcbTest
./pcb_gui_inspeccion.sh
```

Interfazea irekitzean lau fitxa nagusi ikusiko dituzu: **Inspección**, **Rutas**, **Cámara** eta **Configuración de inspección**. Haien funtzionamendua ondorengo ataletan azaltzen da.

> **Oharra:** `gui_config.json` fitxategia ez da beste ordenagailu batera kopiatu behar. Fitxategi horrek makina bakoitzeko bide absolutuak gordetzen ditu.

---

## 4. Rutas fitxa

Fitxa honetan programak behar dituen bideak egiaztatu edo aukeratu behar dira. Lehen aldiz erabiltzean, atal hau da konfiguratu beharreko lehenengoa.

| Eremua | Deskribapena |
|--------|--------------|
| **Modelo YOLO** | YOLO eredua. Gomendatua: `weights/best.pt`. Beste kokapen batean badago, bide absolutua aukeratu. |
| **Carpeta de salida** | Emaitzak gordeko diren karpeta. Ohikoa: `results/gui_pcb_inspection/`. |
| **referenceBoard** | Plaka zuzenaren erreferentzia duen karpeta. |
| **config_homografia.json** | Irudi zuzenduko zabalera eta altuera definitzen ditu. |
| **Serigrafía orientación** | Orientazioa erabakitzeko erabiltzen den serigrafia-irudia. |

### config_homografia.json

Fitxategi honek **ez du** plaka detektatzen. Homografia egin ondoren sortuko den irudiaren tamaina definitzen du. Gutxieneko edukia:

```json
{
  "out_width": 1355,
  "out_height": 774
}
```

> **Kontuz:** Tamaina hau aldatzen bada, `referenceBoard/labels/referencia.txt` fitxategiko koordenatuak irudi zuzenduko tamaina horri lotuta daude.

### referenceBoard karpeta

```
referenceBoard/
├── notes.json
└── labels/
    └── referencia.txt
```

- `notes.json` – klaseen izenak jasotzen ditu.
- `labels/referencia.txt` – plaka zuzeneko osagaien posizioak YOLO formatuan jasotzen ditu.
- `labels/` karpetan **.txt fitxategi bakarra** egon behar da.

---

## 5. Kamera fitxa

Kamera fitxan kamera-iturria aukeratzen da eta kaptura-proba egin daiteke. Hori egitea komeni da ikuskapen osoa martxan jarri aurretik.

**Ohiko aukerak:**

```
0
1
/dev/video0
/dev/video1
/dev/video2
```

Kamera zein gailutan dagoen ikusteko terminalean exekutatu:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

**TEST cámara botoia**

*TEST cámara* botoiak une horretako kaptura bat egiten du eta irudia fitxa berean erakusten du. Kaptura ondo badago, ikuskapen osoa egiteko prest zaude.

Proba-irudia hemen gordetzen da:

```
results/gui_pcb_inspection/camera_test/latest_camera_test.jpg
```

**Bereizmena**

*Ancho cámara* eta *Alto cámara* `0` balioarekin uzten badira, OpenCV-k kameraren bereizmen lehenetsia erabiliko du. Arazoak badaude, probatu:

- `1280 × 720`
- `1920 × 1080`

> **Garrantzitsua:** TEST kamera huts egiten badu, ez hasi ikuskapenarekin. Lehenik kamera-iturria, Docker baimenak eta `/dev/video*` gailuak egiaztatu.

---

## 6. Ikuskapenaren konfigurazioa

Fitxa honetan detekzio eta konparazio parametroak doitzen dira. Balio hauek zuzenean eragiten diete **MISSING**, **MISPLACED** eta **EXTRA** emaitzei.

| Parametroa | Deskribapena |
|------------|--------------|
| **Metodoa** | Homografia egiteko metodoa. Gomendatua: `hough`. |
| **YOLO konfiantza** | Detekzioak onartzeko gutxieneko konfiantza. Balio handiagoak faltsu positibo gutxiago ematen ditu. |
| **Zentro-distantzia** | Erreferentziaren eta detekzioaren zentroen arteko distantzia maximoa. |
| **Distantzia erlaxatua** | Kasuren batean hautagaiak ez baztertzeko erabiltzen den tolerantzia zabalagoa. |
| **EXTRA fallo gisa** | Aktibatuta badago, osagai gehigarri batek plaka MAL markatuko du. |
| **Kaptura muga** | Botoi bidezko erabilerarako normalean `1`. |

**Hasierako balio gomendatuak:**

| Parametroa | Balioa |
|------------|--------|
| Metodoa | `hough` |
| YOLO konfiantza | `0.49` |
| Zentro-distantzia maximoa | `0.035` |
| Zentro-distantzia erlaxatua | `0.060` |
| Kaptura muga | `1` |
| Iraupena | `0` |
| Tartea | `0` |
| EXTRA fallo gisa | desaktibatuta |

> Ez aldatu parametro guztiak batera. Arazo bat badago, aldatu **parametro bakarra** eta berriz probatu. Horrela jakin daiteke zein aldaketak hobetu edo okertu duen emaitza.

---

## 7. Ikuskapena egitea

Ikuskapena egin aurretik, egiaztatu kamera ondo ikusten dela eta plaka osoa agertzen dela. Plakak **ez luke irudiaren ertzak ukitu behar**.

**Prozedura gomendatua:**

1. Ireki GUIa: `./pcb_gui_inspeccion.sh`
2. Joan **Rutas** fitxara eta egiaztatu fitxategi guztiak.
3. Joan **Cámara** fitxara eta sakatu **TEST cámara**.
4. Kaptura zuzena bada, joan **Inspección** fitxara.
5. Jarri plaka kameraren azpian, osorik eta ondo argiztatuta.
6. Sakatu **Analizar placa**.
7. Itxaron emaitza: **PLACA OK** edo **PLACA MAL**.

**Programa barruan gertatzen dena:**

1. Kamera-kaptura egiten da.
2. Plaka detektatzen da.
3. Homografia aplikatzen da.
4. Serigrafiaren bidez orientazioa egiaztatzen da.
5. YOLO inferentzia egiten da.
6. Detekzioak erreferentziarekin alderatzen dira.
7. Irudia, CSVak eta laburpena sortzen dira.

---

## 8. Emaitzak interpretatzea

Ikuskapenaren ondoren programak irudi bat eta hainbat CSV fitxategi sortzen ditu. GUIan normalean `latest_failures.jpg` irudia ikusiko da, hau da, akatsak nabarmentzen dituen irudia.

| Egoera | Esanahia |
|--------|----------|
| **OK** | Erreferentziako osagaia aurkitu da eta posizioa onargarria da. |
| **MISSING** | Erreferentzian espero zen osagai bat ez da balioz detektatu. |
| **MISPLACED** | Klase bereko osagai bat detektatu da, baina posizioa edo geometria ez da nahikoa ona. |
| **EXTRA** | YOLOk detekzio bat egin du, baina ez dago erreferentzian pareko osagairik. |

**Plaka OK edo MAL izateko irizpidea (lehenetsia):**

GUIak plaka zuzena dela erabakitzen du baldintza hau betetzen denean:

- `MISSING = 0`
- `MISPLACED = 0`

EXTRA detekzioak lehenetsita abisu gisa hartzen dira. Konfigurazioan **EXTRA fallo gisa** aktibatzen bada, EXTRA bakar batek ere plaka MAL bihurtuko du.

**Emaitzen karpeta:**

```
results/gui_pcb_inspection/
├── raw/latest_raw.jpg
├── corrected/latest_corrected.jpg
├── overlay/latest_result.jpg
├── overlay_failures/latest_failures.jpg
├── components/latest_components.csv
├── comparison/latest_comparison.csv
├── camera_test/latest_camera_test.jpg
├── debug/
└── summary_realtime.csv
```

---

## 9. Doikuntza gomendatuak

**Faltsu positibo asko badaude**

YOLO konfiantza igo. Horrek detekzio ahulak baztertuko ditu.

```
0.49 → 0.55 → 0.60
```

**Osagai zuzenak MISPLACED gisa agertzen badira**

Zentro-distantzia pixka bat igo. Horrela tolerantzia geometrikoa handiagoa izango da.

```
0.035 → 0.045 → 0.060
```

**Osagai errealak MISSING gisa agertzen badira**

YOLO konfiantza pixka bat jaitsi edo argiztapena eta fokua berrikusi.

```
0.60 → 0.55 → 0.49
```

**Homografia txarra bada**

- Ziurtatu plaka osoa irudian agertzen dela.
- Saihestu distira gogorrak eta itzal handiak.
- Plakak ez dezala irudiaren ertza ukitu.
- Begiratu `debug/homography/` karpetako irudiak.
- Hough metodoa mantendu; normalean robustuena delako ertzak ondo ikusten direnean.

**Orientazioa okerra bada**

- Egiaztatu `keypoints/serigrafia.png` fitxategia ona dela.
- Serigrafiak beti ikusgai egon behar du.
- Ez erabili oso eremu txikia edo distira asko duen serigrafia.
- Begiratu `debug/orientation/` karpetako irudiak.

---

## 10. Ohiko arazoak

**Errorea: kamera ezin da ireki**

Lehenengo egiaztatu kamera sisteman ikusten den:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
```

GUIan probatu beste iturri batzuk: `0`, `1`, `/dev/video0`, `/dev/video1`, `/dev/video2`.

**Docker-ek baimen errorea ematen du**

Erabiltzailea `docker` taldean egon behar da:

```bash
sudo usermod -aG docker $USER
```

Ondoren saioa itxi eta berriz ireki behar da.

**Fitxategiak root gisa sortzen dira**

Script berriek Docker erabiltzaile arruntarekin exekutatzen dute. Hala ere, lehenago root gisa sortutako karpetak badaude:

```bash
sudo chown -R $USER:$USER results .ultralytics .config .cache
```

**`no_valid_candidate_same_class` mezua**

Mezu horrek esan nahi du programa klase bereko detekzioak aurkitu dituela, baina ez duela hautagai baliozkorik aurkitu erreferentziako osagai jakin horrekin parekatzeko. Normalean zentro-distantzia, solapamendua, tamaina edo homografiarekin lotuta dago.

---

## 11. Lizentzia eta erabilera-oharrak

pcbTest-en kode-iturria lizentzia honekin banatzen da:

> **GNU General Public License v3.0 or later**
> `SPDX-License-Identifier: GPL-3.0-or-later`

Horrek esan nahi du programa erabili, aztertu, aldatu eta berriz bana daitekeela, baina banatzen diren bertsio eraldatuek GPLv3 edo bateragarria den lizentzia mantendu behar dute.

Dokumentazioa, irudiak eta azalpen-materiala lizentzia honekin banatzen dira:

> **Creative Commons Attribution-ShareAlike 4.0 International**
> `SPDX-License-Identifier: CC-BY-SA-4.0`

**Erabilera-ohar kritikoak:**

- Programa hau hezkuntza eta prototipatze erabilerarako pentsatuta dago.
- Industria-ingurune batean erabili aurretik, balioztatze sakona egin behar da.
- YOLO ereduaren emaitzak entrenamendu-datuen kalitatearen araberakoak dira.
- Kamera, argiztapena eta plaka-posizionamendua errepikakorrak izan behar dira.
- `best.pt` eredua partekatzean, kontuan hartu entrenamendu-datuen jatorria eta lizentzia.

---

*Laburpena: erabilera egoki baterako, lehenik kamera probatu, ondoren homografia eta orientazio debug irudiak berrikusi, eta azkenik YOLO eta konparazio parametroak pixkanaka doitu.*
