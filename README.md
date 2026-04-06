# Logo Animation

An SVG-driven Manim animation that progressively draws a logo, fills it, rasterizes it into a dot field, and then applies a wave-like swirl effect across the dots.

## What This Project Does

The animation pipeline in `LogoAnimation` performs four main stages:

1. Parses `assets/logo_capital.svg` into path segments.
2. Draws each segment in sequence with construction strokes.
3. Crossfades into a filled logo.
4. Converts the filled shape to a dot field and animates a traveling swirl/pulse motion.

## Tech Stack

- Python 3.13+
- [Manim Community](https://docs.manim.community/) (`manim>=0.20.1`)
- [svgelements](https://github.com/meerk40t/svgelements) (`svgelements>=1.9.6`)
- NumPy (pulled transitively by Manim, used directly in code)

## Project Structure

```text
assets/
    logo_capital.svg
src/
    logo-animation/
        main.py
media/
    ... rendered output files ...
```

## Setup

### Option A: uv (recommended)

```bash
uv sync
```

### Option B: pip + virtual environment

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -U pip
pip install manim svgelements
```

## Render The Animation

From the repository root:

```bash
python -m manim src/logo-animation/main.py LogoAnimation -ql --disable_caching
```

Useful quality presets:

- `-ql` low quality, fast iteration
- `-qm` medium quality
- `-qh` high quality
- `-qk` 4K quality (slow)

Rendered videos are written under `media/`.

## Customize The Look And Motion

Most tunable values live as class attributes on `LogoAnimation` in `src/logo-animation/main.py`.

Common controls:

- `background_color`, `construction_color`, `fill_color`
- `final_stroke_width`, `construction_stroke_width`, `highlight_stroke_width`
- `fit_width_ratio`, `fit_height_ratio`
- `dot_min_spacing` (dot density/performance tradeoff)
- `dot_reveal_radius_scale`, `dot_pulse_amplitude`
- `dot_field_chunk_size` (batch size for vectorized mask calculation)

### Performance Note

Dot field generation is vectorized and chunked. If you decrease `dot_min_spacing`, you get more dots and better detail, but render time and memory use increase significantly.

## Troubleshooting

- If Manim fails due to missing system dependencies (FFmpeg/LaTeX), follow the official Manim installation docs for your platform.
- If no shapes appear, verify `assets/logo_capital.svg` exists and contains path data.
- If dots do not visually resize during updaters, use `scale_to_fit_width(...)`-based sizing (already implemented) instead of relying on `Dot.set_radius(...)` behavior.

## License

See `LICENSE`.
