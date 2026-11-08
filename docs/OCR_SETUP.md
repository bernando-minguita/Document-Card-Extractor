# Enabling local OCR

OCR is **optional**. Without it the application still detects, straightens and
exports every document; you simply pick the document type from the drop-down and
confirm the orientation yourself. With a local OCR engine you additionally get:

| Feature | Without OCR | With OCR |
| --- | --- | --- |
| Document detection / cropping | ✅ | ✅ |
| Automatic type (Passport, Iqama, Vehicle registration, Driver's licence, …) | geometric guess only | keyword match, usually ≥ 0.95 confidence |
| Identifier in the filename (`Passport_WE4110283_P01_01.jpg`) | sequential number | passport number / 10-digit Iqama |
| 180° orientation | amber “Check orientation” badge | resolved when *Use OCR to resolve 180°* is on |
| Speed | — | Tesseract ≈ 0.3–1 s/card, EasyOCR ≈ 7–25 s/card on a CPU |

Everything runs **on your computer**. The application never sends data anywhere
and never downloads anything at runtime.

---

## 1. Choose an engine

| | Tesseract | EasyOCR |
| --- | --- | --- |
| Install size | ~50 MB (+ language data) | ~250–600 MB (PyTorch) |
| Speed per card (CPU) | **0.3–1 s** | 7–25 s |
| Accuracy on ID cards | good | **better** |
| Arabic | needs the `ara` data pack | needs `arabic_g2.pth` |
| Download needed | installer only | installer **+ ~100 MB models** |
| Recommended for | normal use, older laptops | best quality, few documents |

Both can be installed; the application prefers Tesseract (fast) and falls back to
EasyOCR. The engine can be forced in **Settings → OCR engine**
(`Automatic` / `Tesseract` / `EasyOCR`).

---

## 2. Requirements

* Windows 10/11, 64 bit; Python 3.10+ (the interpreter the app runs with).
* The application already installed and starting (`python main.py`).
* Internet access **once** while installing; afterwards everything is offline.

---

## 3. Option A — Tesseract (recommended)

### 3.1 Install the program

1. Download the Windows installer from
   <https://github.com/UB-Mannheim/tesseract/wiki> (`tesseract-ocr-w64-setup-…exe`).
2. In the installer expand **Additional language data** and tick the languages you
   need — e.g. **Arabic** next to the pre-selected *English*.
3. Keep the default folder `C:\Program Files\Tesseract-OCR` (the app finds it
   there automatically). If you install elsewhere, add that folder to `PATH`.

### 3.2 Install the Python wrapper

```bat
pip install pytesseract
```

### 3.3 Verify

```bat
tesseract --version
tesseract --list-langs
python cli.py --check-ocr
```

`--list-langs` must show `eng` and (if ticked) `ara`. `--check-ocr` should print
`Tesseract found : True` and a successful smoke test. No model files are needed —
the language data came with the installer, so a Tesseract setup can be copied to
an offline computer as a folder.

---

## 4. Option B — EasyOCR (best accuracy, needs models)

### 4.1 Install the packages

```bat
pip install easyocr
```

This pulls PyTorch (~200 MB download) plus helper packages. If pip wants to change
`numpy` or `opencv`, accept it — the app supports the newer versions.

> Offline install: `pip download easyocr -d wheels` on a connected PC, then
> `pip install --no-index --find-links wheels easyocr` on the target.

### 4.2 Download the model files **once**, on a machine with internet

EasyOCR fetches its weights the first time a `Reader` is created, so do it
manually (this is the only downloading step):

```bat
python -c "import easyocr; easyocr.Reader(['en','ar'], gpu=False)"
```

That writes `craft_mlt_25k.pth` (text detector), `english_g2.pth` and
`arabic_g2.pth` (one recognition model per language) into:

```
%USERPROFILE%\.EasyOCR\model
```

Only request the languages you need — each recognition model is ~15–45 MB.

### 4.3 Copy the models to the offline computer

Copy the whole `model` folder to the same place on the target machine:

```
C:\Users\<user>\.EasyOCR\model\craft_mlt_25k.pth
C:\Users\<user>\.EasyOCR\model\english_g2.pth
C:\Users\<user>\.EasyOCR\model\arabic_g2.pth
```

Prefer a portable location? Point an environment variable at a folder that
**contains** a `model` sub-folder and the app will use that instead:

```bat
setx EASYOCR_MODULE_PATH "D:\document-card-extractor\easyocr"
```

```
D:\document-card-extractor\easyocr\model\craft_mlt_25k.pth
```

### 4.4 Why this cannot silently download

`core/ocr_engine.py` verifies the files *before* creating the reader and always
passes `download_enabled=False`. If a file is missing, OCR is reported as
unavailable and the settings page shows the folder that is expected — the app
never falls back to fetching anything.

---

## 5. Turn OCR on inside the application

**Settings → Orientation and classification**

1. Tick **Enable local OCR (used for classification and identifiers)**.
2. Optional: tick **Use OCR to resolve 180° orientation (slow)** — only worth it
   with Tesseract or a fast CPU.
3. **OCR engine**: leave `Automatic`, or force `Tesseract` / `EasyOCR`.
4. **OCR languages**: `en`, or `en,ar` for English + Arabic. The ISO 639-1 codes
   are mapped automatically to Tesseract's `eng+ara` notation.
5. Press **Check** next to the engine drop-down: the app runs a small recognition
   test and reports which engine is installed, which one it would use, where the
   EasyOCR models are expected and what the test recognised (or the exact install
   steps when nothing is usable).
6. Press **Save settings** — a confirmation dialog tells you where the settings
   were stored.
7. Press **Start Detection** (or **Re-run Page**).

The sidebar shows what was detected, e.g. `OCR engine: Tesseract` or
`OCR engine: EasyOCR (model files missing)`.

### 5.1 See the text each card produced

Knowing *what* OCR read is the key to fixing a wrong classification.

* **Status bar** after a run: `Detected 4 document(s) · OCR: tesseract (4/4 cards read)`.
* **Gallery tooltip** (hover a thumbnail): the first ~150 characters of the text.
* **Right-click a thumbnail → Show OCR text…**: the full text, with a *Copy text*
  button (paste it into *Edit keywords…* below).
* **Adjust crop… → OCR text**: read-only box with the recognised text and a copy
  button.
* **CLI**: `python cli.py sample.pdf --ocr --show-ocr-text` prints
  `OCR text: …` under every card.

### 5.2 Support another language (even for built-in types)

Built-in types are not hard-coded from the user's point of view any more:

1. Settings → **Document types**. Every type is a tile; built-in types are the
   first ones, your own follow them.
2. Press **Keywords…** on the tile (e.g. *Passport*). The dialog lists the keywords
   the app already knows and offers a free text box for **additional** ones — one
   per line or comma separated, any script:
   ```
   PASSEPORT
   جواز سفر
   TRAVEL DOCUMENT
   ```
3. Save. The words are matched *in addition* to the built-in keywords, so
   classification now also works for those spellings/languages. The tile summary
   shows `your keywords: …`, the tooltip lists built-in and your keywords
   separately, and clearing the box restores the built-in behaviour.
4. The same works for your own types, and the setting is stored in
   `settings.json` (`type_keyword_overrides`) so it survives restarts.

Remember to also install the matching OCR language data (`ara` for Tesseract,
`arabic_g2.pth` for EasyOCR) and to list the code in **OCR languages** —
keywords can only match text the engine actually read.


### Headless (CLI)

```bat
python cli.py sample.pdf --ocr
python cli.py sample.pdf --ocr --ocr-languages en,ar --pages 1
```

---

## 6. Verify the installation

```bat
python cli.py --check-ocr
```

Expected output on a working Tesseract setup:

```
OCR diagnostics
  Configured languages : en
  Tesseract found      : True
  EasyOCR installed    : False
  EasyOCR model folder : C:\Users\<user>\.EasyOCR\model
  Models for en        : False
  Usable engines       : Tesseract
  Smoke test           : engine=tesseract time=0.31s text='PASSPORT WE4110283'
```

If no engine is usable it prints the installation instructions for both engines
instead and exits with code 1.

### 6.1 “Installed” is not the same as “used” — how to confirm

| Where | What you see | Meaning |
| --- | --- | --- |
| **Status bar** right after *Start Detection* | `Detected 4 document(s)  ·  OCR: tesseract (4/4 cards read)` | Tesseract ran and returned text for all 4 cards |
| same, but `(0/4 cards read)` | `OCR: tesseract (0/4 cards read)` | the engine ran but recognised nothing — the crop is too small/blurred, or the wrong language data is installed (not that OCR was skipped) |
| same, no `OCR:` part at all | `Detected 4 document(s)` | OCR is **switched off** — tick *Enable local OCR* in the settings |
| same, `OCR: enabled, but no engine is available` | — | the checkbox is on but neither engine is usable (see §7) |
| **CLI** | `OCR: tesseract (languages: eng)` per PDF | which engine was chosen for this run |
| **CLI**, per card | `... rot=  0° ocr=text` / `ocr=empty` | whether text was read for that card |
| **Settings → Orientation and classification** | hint line *“Detected local OCR engine(s): Tesseract + EasyOCR”* | what is installed on the machine |
| **Sidebar** | `OCR engine: Tesseract` | same information, always visible |

To remove all doubt, force the engine: **Settings → OCR engine → `Tesseract`**
(instead of `Automatic`). If the status bar then reports `OCR: tesseract`, that is
exactly the engine that processed your cards.

Note that classification needs **both** *Enable local OCR* **and**
*Classify documents*; with classification off, OCR is only used if you also enable
*Use OCR to resolve 180° orientation*.


---

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Sidebar shows *“EasyOCR (model files missing)”* | The `.pth` files are not in `%USERPROFILE%\.EasyOCR\model` (or `EASYOCR_MODULE_PATH\model`). Repeat step 4.2/4.3. |
| *“No OCR engine found”* | Neither Tesseract nor EasyOCR is installed. Do option A (simplest) and restart the app. |
| `TesseractNotFoundError` | Tesseract is installed in a non-standard folder. Reinstall to `C:\Program Files\Tesseract-OCR`, or add that folder to `PATH`, then restart the app. |
| Arabic text is not recognised | `ara` data missing: rerun the Tesseract installer and tick *Arabic* (`tesseract --list-langs` must list it), or put `arabic_g2.pth` in the EasyOCR model folder. Check that **OCR languages** contains `ar`. |
| OCR runs but returns empty text | The crop is too small, blurred or upside down. Render at a higher DPI, tighten the crop, or fix the orientation first. |
| EasyOCR is very slow | Normal on a CPU (7–25 s per card). Use Tesseract, or enable OCR only for the pages that need it. |
| `ImportError: DLL load failed` after installing EasyOCR | PyTorch needs the Microsoft Visual C++ Redistributable (2015–2022, x64) — install it and restart. |
| pip changed `numpy` / `opencv` versions | Fine, the app supports the newer versions. If detection misbehaves, reinstall the base set: `pip install -r requirements.txt`. |
| OCR is ignored although a model is installed | **Enable local OCR** is unticked, or the engine is forced to the missing one in **Settings → OCR engine**. Also check the sidebar line. |

---

## 8. What was measured

Reference machine (Windows 11, CPU only, the four documents on `sample.pdf`):

| Engine | Time per card | Passport | Vehicle registration | Driver's licence | Iqama |
| --- | --- | --- | --- | --- | --- |
| none | — | Passport 0.22 (layout hint) | Unknown | Unknown | Unknown |
| EasyOCR (`en`) | 7–25 s | Passport 0.95 | **0.99** | **1.00** | 0.55 |

OCR output only feeds the classifier — a wrong classification never blocks the
export; correct the type in the drop-down and re-export.


