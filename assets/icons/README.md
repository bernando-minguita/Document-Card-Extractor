# Icons

The application icons are **vector icons generated at runtime** from the embedded
SVG path definitions in `app/icons.py` (`icon(name, color)` renders them with
`QSvgRenderer` at several sizes, so they stay crisp on every DPI and can be
recoloured for normal/hover/selected states).

This folder is kept so that binary icon assets (`.ico`, `.png`) can be dropped in
later - for example a custom application icon for PyInstaller:

```bat
python -m PyInstaller --windowed --icon assets\icons\app.ico main.py
```

Names currently available in `app/icons.py`: `dashboard`, `extract`, `history`,
`settings`, `folder`, `plus`, `play`, `export`, `save`, `zoom_in`, `zoom_out`,
`fit`, `rotate_left`, `rotate_right`, `flip`, `trash`, `edit`, `crop`, `check`,
`close`, `chevron_left`, `chevron_right`, `image`, `alert`, `refresh`, `search`,
`document`, `drag`.
