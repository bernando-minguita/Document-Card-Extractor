# Document Card Extractor

A modern, fully offline Windows desktop application that finds the individual
documents on a scanned PDF page **or image** (passports, Saudi Iqama, vehicle
registrations and driving licences), straightens them and exports every document
as its own image file.

Built with **PySide6**, **OpenCV**, **PyMuPDF**, **Pillow** and **NumPy**.
Detection, orientation correction and export all run locally - no internet, no
cloud services.

**Accepted input:** PDF, PNG, JPG/JPEG, TIFF (including multi-page TIFF from a
document scanner), BMP, WEBP. Every frame of a multi-page TIFF becomes a page, so
image scans behave exactly like PDFs (page navigation, detection, export).

---

## 1. What it does

1. You open (or drag & drop) a scanned PDF or image.
2. **Start Detection** analyses every selected page and finds the document cards.
3. Each card is perspective corrected, its orientation is analysed and it is
   classified (passport / Iqama / vehicle registration / driving licence / unknown).
4. You review the thumbnails, rotate or re-crop anything that needs it, assign the
   document type and exclude whatever you do not want.
5. **Export All** writes one image per approved document into
   `output/<file name>/`, using the naming scheme
   `DOCUMENTTYPE_IDENTIFIER_PAGENUMBER_SEQUENCE`.

A single page can contain landscape and portrait documents at the same time, at
different sizes and positions, slightly rotated or skewed.

---

## 2. Installation

Windows 10/11, Python 3.10+ (64 bit). The dependencies are installed into the
interpreter's normal `site-packages` directory - no virtual environment and no
special `sys.path` configuration is needed.

### 2.1 Install the dependencies

```bat
pip install -r requirements.txt
python main.py
```

Alternatively double-click `run.bat`, which starts `main.py` with the Python on
your `PATH`.

### 2.2 Optional local OCR

Everything works without OCR — detection, cropping, export and manual type
selection. Installing one engine additionally enables automatic classification,
document identifiers in the filename and reliable 180° orientation detection.

> **Full step-by-step guide: [docs/OCR_SETUP.md](docs/OCR_SETUP.md)** — requirements,
> install commands, how to download the EasyOCR models, verification and a
> troubleshooting table.

Quick summary:

| Engine | Install | Speed | Notes |
| --- | --- | --- | --- |
| **Tesseract** (recommended) | installer from <https://github.com/UB-Mannheim/tesseract/wiki> (tick *Arabic* if needed) + `pip install pytesseract` | 0.3–1 s/card | language data ships with the installer, nothing to download later |
| **EasyOCR** | `pip install easyocr` | 7–25 s/card | needs the ~100 MB model files, see below |

```bat
:: install both wrappers (either engine is enough)
pip install -r requirements-ocr.txt
:: verify what the application can use
python cli.py --check-ocr
```

#### EasyOCR works offline and needs pre-downloaded models

The application passes `download_enabled=False`, i.e. it **never** downloads
anything at runtime. EasyOCR's weights (a CRAFT text detector plus one
recognition model per language, ~100 MB for English) must therefore already be on
the computer:

1. On a machine **with** internet access, run once:
   ```bat
   python -c "import easyocr; easyocr.Reader(['en','ar'], gpu=False)"
   ```
   This writes `~/.EasyOCR/model/` (`craft_mlt_25k.pth`, `english_g2.pth`,
   `arabic_g2.pth`, …).
2. Copy that folder to the offline computer to the same place
   (`%USERPROFILE%\.EasyOCR\model`), or point the environment variable
   `EASYOCR_MODULE_PATH` at a folder that contains the `model` sub-folder.

If the models are missing the app says so instead of failing silently: the
sidebar shows **"EasyOCR (model files missing)"**, OCR is treated as unavailable
and the settings page + `python cli.py --check-ocr` explain what is missing. The
built-in document types still work by hand, and Tesseract (if installed) is used
instead.

The sidebar shows which OCR engine was detected.

---

## 3. Running

| Command | Purpose |
| --- | --- |
| `python main.py` | Start the desktop application |
| `python main.py scan.pdf` | Start and open the given PDF immediately |
| `python main.py scan1.pdf scan2.png` | …open several files (each becomes a document in the top-right drop-down) |
| `python cli.py scan.pdf` | Headless batch extraction (no GUI); accepts several files at once |
| `python cli.py scan.png` | …or an image file (multi-page TIFF supported) |
| `python cli.py --check-ocr` | Report which local OCR engine/models are ready |
| `python cli.py scan.pdf --pages 1,3 --format png --dpi 400 --debug-dir _debug` | Selected pages, PNG, debug overlays |
| `python cli.py scan.pdf --ocr` | Batch extraction with OCR classification |
| `python -m pytest tests -q` | Run the automated test suite |

`cli.py --help` lists every option (sensitivity, working resolution, output
folder, flat export, …).

---

## 4. Using the application

### Workflow

1. **Open File** (or drag & drop it onto the window, or **Add File** to keep several
   PDFs/images side by side - the drop-down at the top right switches between them;
   dropping several files at once opens them all). PDFs and images are both
   accepted; a multi-page TIFF is treated as a multi-page document.
2. Navigate with the page spinner / arrows. The page is rendered at the preview DPI
   and detected cards are drawn as coloured polygons.
3. **Start Detection** processes all pages, **Re-run Page** only the current one.
4. Click a card (in the page view or in the gallery) to select it, then:
   * drag the four round handles to correct the crop,
   * drag **inside** the selected card to move the whole region (all four
     corners together) without changing its shape,
   * use the rotate buttons (90° left / right / 180°),
   * open **Adjust crop…** (double click) and drag the **round grip above the
     crop** to rotate it freely - the *Fine rotation* field follows the drag
     (0.1° precision, ±45°, and beyond that the angle is folded into the crop and
     rotation continues), so a slightly tilted scan can be levelled by hand,
   * fine-tune the crop without the mouse: **arrow keys** move it by 1 px,
     **Shift+arrows** make it wider/narrower/taller/shorter,
     **Ctrl+arrows** use 10 px steps (there are `◀ ▲ ▼ ▶` and `− W / + W / − H / + H`
     buttons in *Adjust crop…* that do the same),
   * use the **Margin** field in *Adjust crop…* ("Add to all sides", e.g. 2 mm) to
     keep the shadow and the rounded corners of the card inside the export,
   * pick the document type in the drop-down,
   * untick **Include** to keep it out of the export.
5. **Add Region** lets you draw a rectangle for a document the detector missed
   (`Esc` cancels). The drop-down next to it sets the size of that region:
   *Free (use the drawn box)*, an **ID card 85.6 × 54 mm**, a business card
   (90 × 55 mm), a passport page (125 × 88 mm) or **Custom…**, which opens a small
   dialog for your own width/height (with quick presets and a swap button).
   The button beside the drop-down always shows the active size (or *Size…* when
   free drawing is on) and re-opens that dialog at any time - even when "Custom…"
   is already the selected entry.
   With a fixed size the region keeps the centre of your drag and takes exactly
   that physical size, so an identity card is cropped precisely - no guessing
   where its edges are. These controls stay available even before a PDF is opened.
6. **Export All** exports every approved document of every loaded PDF. Each PDF
   gets its own sub-folder.

The status bar shows pages, detected and approved documents, the current activity
and a cancel button for long running work. After detection it also reports which
OCR engine processed the cards, e.g.
`Detected 4 document(s)  ·  OCR: tesseract (4/4 cards read)` — see
[docs/OCR_SETUP.md](docs/OCR_SETUP.md#61-installed-is-not-the-same-as-used--how-to-confirm).

**Extraction history** — every export is recorded on the *Extraction History*
page (input file, document count, output folder, status). Double-click an entry
(or *Open output folder*) to jump back to the exported images; *Refresh* rereads
the file, *Clear history* removes the entries without touching the exported
files. It lives next to the settings in your user profile.

### Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| `Ctrl+O` | Open PDF |
| `Ctrl+R` | Start detection |
| `Ctrl+E` | Export all |
| `Ctrl` + mouse wheel | Zoom the page |
| `Esc` | Leave "Add Region" mode |
| `←` `→` `↑` `↓` | Move the selected crop by 1 px (in *Adjust crop…* and on the page view) |
| `Shift` + arrows | Resize the crop: `→`/`←` wider/narrower, `↑`/`↓` taller/shorter (1 px per side) |
| `Ctrl` + arrows | Same, but 10 px per press (`Alt` = 20 px) |

### Output files

```
output/
└── sample/                      <- one folder per input file (configurable)
    ├── Passport_WE4110283_P01_01.jpg
    ├── Iqama_2567251562_P01_02.jpg
    ├── VehicleRegistration_003_P01_03.jpg
    └── DriversLicense_004_P01_04.jpg
```

The default pattern is `{type}_{identifier}_{page}_{sequence}`. Placeholders:
`{type}`, `{identifier}`, `{page}` (`P01`), `{sequence}` (`01`), `{pdf}`. When no
identifier is available a sequential 3-digit number is used. Existing files are
never overwritten - a `_2`, `_3`, … suffix is appended.

---

## 5. How the detection works

```
PDF page ──render (300 dpi, capped to the native scan resolution)──► BGR image
image file ──decode (native pixels, EXIF rotation applied)──────────► same pipeline
   │
   ├─ working copy (max. 2200 px); all geometry is stored normalised, so the
   │  export can re-render the page at full resolution later
   │
   ├─ pre-processing
   │    • grey conversion, light denoising
   │    • flat-field illumination correction (removes scanner shading/shadows)
   │    • CLAHE contrast, Scharr gradient magnitude
   │
   ├─ six independent candidate strategies
   │    1. adaptive threshold (both polarities)
   │    2. Otsu threshold (both polarities)
   │    3. edge outline: Canny → dilate → fill holes → erode   (best for light
   │       cards on a light platen; never bridges the gap between two cards)
   │    4. gradient magnitude with the same fill pipeline
   │    5. colour / saturation mask for coloured cards
   │    6. connected components of all the above
   │
   ├─ contour → polygon approximation → rectangle validation
   │    area, side length, aspect ratio, rectangularity, extent, edge support,
   │    "not the whole page", convexity
   │
   ├─ scoring: edge support 30 %, rectangularity 20 %, area 18 %, aspect 18 %,
   │            extent 14 %, multiplied by a per-strategy weight
   │
   ├─ duplicate removal / nesting resolution
   │    • IoU filter removes the same card found by several strategies
   │    • printed frames and photo blocks inside a card are dropped in favour of
   │      the physical card edge
   │    • a rectangle that is really two touching documents (two disjoint inner
   │      detections covering most of it) is split instead of merged
   │
   ├─ quad refinement: each edge is snapped onto the strongest nearby gradient
   │    (line fitting + intersection), guarded by convexity/area checks
   │
   └─ padding: the quad is expanded by 0.6 % of its diagonal so the physical
      card edge is never cut off
```

Each accepted card is perspective-warped (`cv2.getPerspectiveTransform` +
`warpPerspective`, cubic interpolation, `BORDER_REPLICATE`) and stored as a
normalised quad plus a thumbnail.

### Orientation detection

* **Heuristic (always on, milliseconds per card):** the shape of the word blobs is
  measured. Upright text produces blobs wider than tall, text rotated by 90°
  produces taller than wide blobs. This reliably separates `0°/180°` from
  `90°/270°`.
* The `0°` versus `180°` flip cannot be decided without reading the text, so the
  card is marked *"Check orientation"* (amber badge) instead of guessing - one
  click on the 180° button fixes it.
* With OCR enabled and *"Use OCR to resolve 180° orientation"* switched on, the text
  is recognised in both candidate rotations and the readable one wins.

### Classification

Classification is advisory. It combines OCR keywords (`PASSPORT`,
`RESIDENT IDENTITY`, `VEHICLE REGISTRATION`, `DRIVING LICENSE`, their Arabic
equivalents and the MRZ pattern) with weak layout heuristics. Low confidence
results in `Unknown document`; the card is never rejected, you just pick the type
from the drop-down. Identifiers (passport number, 10-digit Iqama number) are
extracted from the OCR text and used in the filename.

The keyword match is OCR-noise tolerant: the text is upper-cased, common OCR
confusions are normalised (`0→O`, `5→S`, `8→B`, `(`→`C`, `$→S`, …) and words are
compared with a similarity ratio, so `(EHICLE $ REBISTAATION` still matches
`VEHICLE REGISTRATION` and `REBISTAATION` matches `REGISTRATION`.

Measured on the bundled sample scan (EasyOCR, English):

| Card | without OCR | with OCR |
| --- | --- | --- |
| open passport | Passport 0.22 (layout hint) | Passport 0.95 |
| vehicle registration | Unknown | **Istimara 0.99** |
| driving licence | Unknown | **Driver's License 1.00** |
| resident identity | Unknown | **Iqama 0.55** |

OCR costs roughly 7–25 s per card on a CPU, which is why it is off by default.

### Adding your own document types

Settings → **Document types** shows every type as a **tile** (two per row), each with a
checkbox, its category and its keywords:

| Type | Category | Removable |
| --- | --- | --- |
| Passport, Iqama, Vehicle Registration, Driver's License, Health Insurance Card, Bank Card, Salary Card, National ID, Unknown | built-in | no (untick to hide) |
| anything you add | custom | yes (`Remove` on the tile) |

* **Unticking** a type hides it from the document type pickers *and* from
  automatic classification; the tile greys out. Existing cards keep their type.
  `Unknown` is always available and cannot be unticked.
* **Add type…** asks for a name and (optionally) the words that appear on that
  document. The new type immediately appears in the gallery drop-down and in the
  *Adjust document* dialog, is used for the filename
  (`MembershipCard_001_P01_01.jpg`) and — when it has keywords and OCR is enabled
  — is assigned automatically.
* **Remove** only appears on your own types; built-in types can only be hidden.
* **Keywords…** is on *every* tile, including the built-ins: the dialog shows the
  keywords the app already knows and lets you add your own (one per line or comma
  separated, any language/script). They are matched in addition to the built-in
  keywords, so `PASSEPORT`, `جواز سفر` or `TRAVEL DOCUMENT` can be added to
  *Passport* without touching the code. Clear the box to return to the built-in
  keywords.

The state lives in `settings.json` (`custom_document_types`,
`disabled_document_types`, `type_keyword_overrides`). The built-in list is defined
in `core/models.py` (`DocumentType.register(...)`), the auto-classification
keywords in `core/document_classifier.py`.

### Seeing what OCR found

Classification is only as good as the text it receives, so the recognised text is
available everywhere:

* the **status bar** after detection reports the engine and how many cards it read,
* the **gallery tooltip** shows the first characters, **right-click → Show OCR text…**
  the full text (with a copy button),
* **Adjust crop…** has an *OCR text* box,
* `python cli.py scan.pdf --ocr --show-ocr-text` prints it per card,
* `python cli.py --check-ocr` verifies the engine itself. Settings also has a
  **Check** button next to *OCR engine* that runs the same test inside the app.

Full guide: [docs/OCR_SETUP.md](docs/OCR_SETUP.md).


---

## 6. Settings

Stored in the user profile (`%APPDATA%\DocumentCardExtractor\settings.json`, or
`~/.config/DocumentCardExtractor/settings.json` on Linux) and editable on the
Settings page.

| Setting | Default | Meaning |
| --- | --- | --- |
| Render / detection DPI | 300 | resolution used for detection and export |
| Never upscale scans | on | caps the DPI to the native resolution of the scan |
| Preview DPI | 130 | resolution of the page preview |
| Detection sensitivity | 5 | 1 = strict, 10 = also finds faint edges |
| Minimum / maximum document area | 0.8 % / 92 % | detection gates |
| Minimum confidence | 0.42 | cards below this score are ignored |
| Working resolution | 2200 px | downscale for detection (speed vs. detail) |
| Snap edges to gradients | on | quad refinement |
| Fixed size for Add Region | off | drawn regions are snapped to a physical size (e.g. ID card 85.6 × 54 mm) |
| Add Region preset | Free | Free / ID card / Business card / Passport page / Custom (asks for the size) |
| Add Region width / height | 85.6 × 54 mm | the size used when a fixed size is selected (10-400 mm) |
| Automatic orientation | on | heuristic rotation |
| Automatic classification | on | keyword / geometry classification |
| Enable local OCR | off | enables OCR based classification |
| Use OCR for 180° orientation | off | slow, but resolves the flip |
| OCR engine | Automatic | Automatic / Tesseract / EasyOCR (with a **Check** button that tests it) |
| OCR languages | en | `en`, `en,ar`, … mapped to `eng+ara` for Tesseract |
| Image format | JPG | JPG / PNG / TIFF |
| JPEG quality | 95 | 60-100 |
| Output folder | `<project>\output` | where images are written |
| Separate sub-folder per PDF | on | `output/<pdf name>/` |
| Filename format | `{type}_{identifier}_{page}_{sequence}` | see above |
| Open output folder after export | on | Explorer opens when done |
| Document types | 9 built-in | add your own on the Settings page (name + optional OCR keywords) |

---

## 7. Project structure

```
main.py                     entry point (opens every scan given on the command line)
cli.py                      headless batch extraction (several files per run)
requirements.txt
requirements-ocr.txt        optional OCR extras (Tesseract wrapper, EasyOCR)
README.md
docs/OCR_SETUP.md           how to enable local OCR and download the models
run.bat                     launcher (starts main.py with the Python on PATH)
sample.pdf                  local sample scan (4 documents); git-ignored, the
                            tests need it only for the accuracy tests

app/
  main_window.py            main window, workflow, Add Region state machine,
                            document sessions
  settings_page.py          settings page: forms, save/restore, document type manager
  history_page.py           extraction history page
  settings.py               AppSettings dataclass + JSON persistence (atomic writes)
  history.py                export history (JSON, atomic writes)
  session.py                one loaded PDF and its detected cards
  styles.py                 light theme palette + global style sheet
  icons.py                  vector icons (generated from embedded SVG)

core/
  pdf_loader.py             document loader: PDF (PyMuPDF) and image (Pillow) pages,
                            DPI aware rendering, errors
  image_preprocessor.py     illumination correction, masks, hole filling
  card_detector.py          multi-strategy detector, scoring, merging, refinement
  perspective_corrector.py  quad ordering, warping, rotation, IoU helpers
  orientation_detector.py   word-shape heuristic + optional OCR refinement
  document_classifier.py    keyword/geometry classification, identifier extraction
  ocr_engine.py             optional Tesseract / EasyOCR backend
  image_exporter.py         filenames, DPI aware saving (JPG/PNG/TIFF)
  models.py                 DocumentType, DetectedCard, quad helpers

workers/
  detection_worker.py       QThread: render → detect → warp → orient → classify,
                            cancellable between stages
  export_worker.py          QThread: re-render pages, warp, save

widgets/
  pdf_viewer.py             QGraphicsView page view with draggable card quads,
                            drag-to-move, rotation grip and manual rectangles
  document_gallery.py       thumbnail gallery with context menu
  crop_editor.py            large crop / orientation editor dialog
  region_size_dialog.py     width/height dialog for "Add Region → Custom…"
  progress_widget.py        status bar progress + cancel
  qt_image.py               numpy ↔ Qt image conversion helpers

tests/
  conftest.py               generates a synthetic multi-card PDF so most tests
                            run without sample.pdf
  test_detection.py         engine tests on the bundled sample scan
  test_units.py             geometry, filenames, classification, orientation,
                            settings/history persistence
  test_workers.py           detection/export worker behaviour (cancel, sequence,
                            error handling, GUI-token snapshots)
  test_gui.py               end-to-end GUI test (offscreen)

assets/icons/               note about the vector icons
output/                     default export folder
```

### Performance and memory

* Detection and export run in `QThread` workers - the UI never blocks and both can
  be cancelled.
* Only the current page is rendered for the preview; pages are never all kept in
  memory. Cards are stored as normalised quads plus a small thumbnail, and the
  export re-renders each page once at full resolution.
* One page takes roughly 1.5-3 s (2200 px working resolution, 4 documents). Add
  ~1-3 s per document if OCR is enabled (EasyOCR, CPU).

---

## 8. Error handling

The application reports instead of crashing for: unreadable/corrupted PDFs or
images, unsupported file types, password protected PDFs (it asks for the
password, keeps it per document for the whole session so detection and export
never have to ask again, and re-prompts if the password was wrong), pages that
fail to render, pages without any document, very low resolution scans, a failing
page inside a batch, export permission problems and per-file export errors.
Detection warnings are shown after the run and successful files are still
exported. Settings and history are written atomically and both survive a crash.

---

## 9. Tests

```bat
python -m pytest tests -q
```

* `tests/conftest.py` – generates a synthetic multi-card PDF, so nearly the whole
  suite runs **without** `sample.pdf` being present.
* `tests/test_detection.py` – four documents found on the bundled `sample.pdf`,
  complete crops, no duplicates, blank pages stay empty, a synthetic card is
  located within a few pixels, and a **PNG image** is detected as a one-page
  document.
* `tests/test_units.py` – quad ordering, warping, rotation (incl. fine rotation),
  resize/expand along the quad's own axes, IoU, filenames, model serialisation,
  classification, orientation, image/PDF loading (multi-page TIFF, unsupported and
  broken files), settings/history persistence (atomic writes, corrupt-file
  recovery), document type registration.
* `tests/test_workers.py` – the detection/export worker threads run synchronously:
  file sequence numbering, cancellation, per-page error reporting, GUI edit
  snapshots (settings and cards are frozen when the threads start).
* `tests/test_gui.py` – offscreen `MainWindow` test: open PDF **and image** →
  detect → edit (handles, drag to move, keyboard nudge/resize, margin, fine
  rotation, Add Region with a fixed card size) → export → history entry, plus the
  crop editor, the settings page and the document type manager.

---

## 10. Packaging (optional)

```bat
pip install pyinstaller
python -m PyInstaller --noconfirm --windowed --name "Document Card Extractor" ^
    --add-data "assets;assets" main.py
```

The executable is written to `dist\Document Card Extractor\`.

---

## 11. Publishing to GitHub

`.gitignore` is already set up. The important part: **scanned PDFs are excluded**,
because they usually contain personal data (passports, ID cards, licences):

```
*.pdf
```

`sample.pdf` is therefore *not* committed and `python -m pytest tests` skips the
sample-scan tests automatically when it is missing. If you want to ship a
sanitised sample, force-add it (`git add -f sample.pdf`) or put it under
`docs/` (that folder is whitelisted).

```bat
git add -A
git commit -m "Document Card Extractor"
git remote add origin https://github.com/<you>/document-card-extractor.git
git push -u origin main
```

(Already done in this working copy: `git init`, the first commit and a local
`user.name` / `user.email`; only `git remote add` + `git push` remain for
publishing.)

Also ignored: `output/`, `_debug/`, scratch files, `__pycache__/`, `.pytest_cache/`,
virtual environments, IDE folders and `settings.json` / `history.json` (they live
in your user profile anyway).

---

## 12. Tips for best results

* Scan at 200-400 dpi. Higher resolution is capped automatically and does not help.
* Leave a small gap between the cards on the platen - it makes splitting trivial.
* If a card is missed, raise the detection sensitivity or use **Add Region**; pick
  the *ID card 85.6 × 54 mm* preset so the region is exactly card sized.
* If a card is slightly tilted, select it and drag the round **rotation grip**
  above the crop in *Adjust crop…* (or type an angle in *Fine rotation*).
* If two cards were merged into one, the detector usually splits them; if not,
  remove that region and add both cards manually.
* Always check the amber *"Check orientation"* hint before exporting - a 180° flip
  cannot be detected without OCR.



