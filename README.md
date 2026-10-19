# Nordkuppel observations

Pipeline for reducing and combining CCD observations taken at the Nordkuppel telescope
(FLI camera, 2048 × 2048 px at 2 × 2 binning, narrow-band Hα / OIII / SII filters) into one
SHO colour image.

## Contents

| File | Purpose |
| --- | --- |
| `postprocess.sh` | The full pipeline: reduction, astrometry (SCAMP), stacking (SWarp) and SHO colour composite (STIFF) |
| `Nordkuppel_Reduction.py` | Bias, dark and flat calibration of all filters of one night, with optional cosmic-ray removal and background subtraction; called by `postprocess.sh` |
| `prepare_composite.py` | Crops the per-filter stacks to the area covered well in every filter and chooses the colour levels; called by `postprocess.sh` |
| `environment.yml` | micromamba/conda environment with the astromatic programs |
| `sextractor.param`, `gauss_2.5_5x5.conv` | SExtractor output parameters and a 5 × 5 Gaussian filter kernel (FWHM 2.5 px) |

## Data layout

Raw data are kept per target and night and are not tracked in git:

```
NGC7380/
├── 20220712/
│   ├── 20220712-0001_bias.fit
│   ├── 20220712-0001_dark.fit
│   ├── 20220712-0001_flat_ha.fit
│   ├── 20220712-0001_light_ha.fit
│   └── Reduced/          ← created by the reduction
├── 20220719/
│   ├── 20220719-0001_ha_light.fit
│   ├── 20220719-0001_ha_sky.fit
│   └── ...
└── stack/                ← created by postprocess.sh
Exoplanetentransit/
└── TOI2040/
```

The light-frame naming changed between nights (`light_ha` vs. `ha_light`), and the SII flats are
called `flat_sii` while the SII lights use `s2`; the reduction accepts both. The `*_sky.fit`
frames are not used.

## Setup

The Python reduction uses [uv](https://docs.astral.sh/uv/) with Python 3.14:

```sh
uv sync
```

The stacking uses the [astromatic](https://www.astromatic.net) programs SExtractor, SCAMP, SWarp
and STIFF. They are not in Homebrew, but conda-forge has them; install them into their own
environment with [micromamba](https://mamba.readthedocs.io):

```sh
brew install micromamba
micromamba create -f environment.yml
```

`postprocess.sh` activates the `astromatic` environment by itself; without micromamba it looks for the
programs on `PATH`.

## Running the pipeline

Give the output directory first, then every night to include:

```sh
./postprocess.sh NGC7380/stack NGC7380/20220712 NGC7380/20220719
./postprocess.sh NGC7380/stack_20220712 NGC7380/20220712           # a single night
```

The output directory (`stack` is only a name; any directory works and is created if needed)
receives everything after the reduction: the linked frames with their catalogues and SCAMP
headers in `frames/`, the per-filter stacks `coadd_<filter>.fits` with weight maps, the crops,
the logs and the colour image. The reduced frames themselves go to `<night>/Reduced/`. Each full
run clears `frames/`, so reusing a directory for a different set of nights does not mix them, but
it does overwrite the previous stacks; use a separate directory per combination to keep them.

The steps are

1. **Reduction** of every night with `Nordkuppel_Reduction.py`, written to `<night>/Reduced/`
   (existing files there are overwritten)
2. link every `reduced_*.fit` into `<out>/frames/` as `<date>-<number>_<filter>.fit`, with its
   night's bad-pixel mask next to it as `<date>-<number>_<filter>.weight.fits`
3. SExtractor: a FITS_LDAC catalogue per frame, ignoring the masked pixels
4. SCAMP: astrometric solution against Gaia (downloaded from VizieR, so it needs network access)
   and relative flux scaling, written as a `.head` file next to each frame
5. SWarp: one common pixel grid for all filters, then `coadd_<filter>.fits` per filter; masked
   pixels get no weight, and each frame is weighted by its noise; the sky is removed as one
   constant level per frame so that the extended nebula is kept
6. `prepare_composite.py`: crop the stacks to `crop_<filter>.fits` (see "Image size" below),
   choose the black and white point of each colour channel and, optionally (on by default), make
   the stars white
7. STIFF: SHO composite `SHO.tif` (R = SII, G = Hα, B = OIII), 16 bits per channel

### Image size

The telescope pointing wanders by up to a frame width between exposures, so the stack is an
irregular patchwork: a deep core where most frames overlap and shallower surroundings covered by
only a few. All three filters cover the same area. Step 6 decides which part of it ends up in the
image; everything else in the bounding rectangle is filled with black. There are three modes:

**`CROP_SHAPE=frames` (default)** builds the shape from the frames themselves: every frame's
outline is projected onto the stack through its SCAMP solution, and every pixel covered by at
least `MIN_FRAMES` frames in every filter is kept, so the edge consists only of straight frame
edges. With `SINGLE_REGION=true` (default) only the connected region around the core is kept, so
detached slivers where a few frames happen to overlap at the edge are dropped. For the two
NGC 7380 nights (up to 40 frames per filter overlap in the core):

| `MIN_FRAMES` | area with data | noise at the edges |
| --- | --- | --- |
| 15 | 4.8 Mpx | 1.6× |
| 10 | 6.5 Mpx | 2× |
| 5 (default) | 8.6 Mpx | 2.8× |
| 1 (union of all frames) | 16.9 Mpx | 6.3× |

The noise grows as √(deepest count / count). With very few frames the outer parts are not only
noisier: a single frame's sky level also does not match the deep stack exactly, so it shows as a
lighter patch.

**`CROP_SHAPE=footprint`** keeps every part that reaches `MIN_WEIGHT` of the full depth in all
filters, judged from the weight maps (default 0.25: at most twice the noise of the core; 0.5 gives
3.5 Mpx, 0.15 gives 7.6 Mpx). Its edge follows a depth contour; `FOOTPRINT_SMOOTHING` passes of a
majority filter remove isolated specks and holes along it.

**`CROP_SHAPE=rectangle`** keeps only the largest rectangle that reaches `MIN_WEIGHT` (with 0.5:
2.45 Mpx), without black areas.

```sh
MIN_FRAMES=10 RGB_NAME=SHO_frames10.tif ./postprocess.sh --render NGC7380/stack
CROP_SHAPE=footprint RGB_NAME=SHO_footprint.tif ./postprocess.sh --render NGC7380/stack
CROP_SHAPE=rectangle MIN_WEIGHT=0.5 RGB_NAME=SHO_rect.tif ./postprocess.sh --render NGC7380/stack
```

The script checks that every program is available before it starts and stops at the first
error. SExtractor's messages go to `<out>/sextractor.log`; warnings other than the harmless ATLAS
one are printed. The conda-forge SCAMP cannot draw check plots, so they are switched off.

### Tuning the colour image

The colour levels are chosen so that the three filters contribute equally, although Hα is about
four times brighter than SII and OIII:

- **black point** of each channel = its level in the darkest 5 % of the field, so dark dust lanes
  and sky come out neutral instead of tinted
- **white point** of each channel = black point + `WHITE_SCALE` × (its 90th percentile − black
  point), i.e. set by the nebula instead of the stars

The knobs are `WHITE_SCALE` (lower is brighter, default 3), `STIFF_GAMMA_FAC` (above 1 lifts
the faint nebula) and `STIFF_COLOUR_SAT`. To try values, re-render only the colour image from the
existing stacks, which takes a few seconds instead of a full run, and give the values on the
command line, so `postprocess.sh` itself stays unchanged:

```sh
./postprocess.sh --render NGC7380/stack                                      # settings as in postprocess.sh
WHITE_SCALE=2.5 ./postprocess.sh --render NGC7380/stack                      # brighter
WHITE_SCALE=2.5 STIFF_GAMMA_FAC=1.2 RGB_NAME=SHO_try.tif ./postprocess.sh --render NGC7380/stack
```

The last line writes to its own file, so the versions can be compared side by side. When you
like a setting, put it into the SETTINGS block.

### Star colours

Stars are about equally bright in all three filters, but SII and OIII are stretched about 4× harder
than Hα, so without correction stars come out pink. With `NEUTRAL_STARS=true` (the default)
`prepare_composite.py` finds the stars — detections whose stretched SII and OIII flux exceeds their
Hα flux by at least `STAR_MIN_EXCESS` (stars have about 4, nebula structure 1 or less) — and,
inside a soft mask around each, makes their light the same in all channels. Only the light above
the local background is changed, and only where it is clearly above the noise, so the nebula
colour under and around the stars is kept. The mask is written as `star_mask.fits` for
inspection.

```sh
NEUTRAL_STARS=false ./postprocess.sh --render NGC7380/stack          # original star colours
STAR_STRENGTH=0.5 ./postprocess.sh --render NGC7380/stack            # halfway to neutral
STAR_MASK_SCALE=3.5 ./postprocess.sh --render NGC7380/stack          # larger masks, for big halos
```

Very faint stars below the detection threshold (`STAR_THRESHOLD`, 3σ) keep a slight tint.

## Settings

Every setting of the pipeline is in the SETTINGS block at the top of `postprocess.sh`, grouped into
reduction, astrometry and stacking, and colour composite. `postprocess.sh` passes the reduction and
composite settings on to the Python steps as command-line options, so there is nothing to edit
in the Python files. Adjust `WCS_CENTER` and `WCS_CD` if the target is not NGC 7380.

Any setting can also be given for a single call by putting it in front of the command, which
leaves the file unchanged, e.g. `COSMICS_REMOVAL=true ./postprocess.sh NGC7380/stack NGC7380/20220712
NGC7380/20220719`. Settings with several values are written in quotes, e.g.
`WCS_CENTER="341.77 58.08"`.

## Reduction

`Nordkuppel_Reduction.py` can also be run on its own; every setting is then a command-line
option with the same default as in `postprocess.sh`:

```sh
uv run python Nordkuppel_Reduction.py NGC7380/20220719
uv run python Nordkuppel_Reduction.py NGC7380/20220719 --filters ha s2 --output NGC7380/20220719/Reduced_v2
uv run python Nordkuppel_Reduction.py NGC7380/20220719 --cosmics-removal --read-noise 19
uv run python Nordkuppel_Reduction.py --help
```

The options are checked before anything runs: an unknown filter, a negative threshold or an even
background filter size is rejected with an error message. The script also stops with an error if
any file pattern matches no file.

Per night, master bias and dark are built once and every filter is reduced with them:

1. master bias = combined stack of all bias frames
2. master dark = combined stack of (dark − bias) / exposure time, i.e. a dark-current rate
3. master flat = combined stack of (flat − bias − dark · t), each normalised to its median
4. bad-pixel mask per filter: hot pixels (dark current more than `HOT_SIGMA` = 10 σ above the
   median; about 0.7 % of the sensor) and pixels whose flat response is outside 0.5–1.5. Hot
   pixels cannot be dark subtracted exactly; left in, they become specks and, after resampling,
   small rings in the stack. The mask is written as the weight map `mask_<filter>.fits`
   (0 = bad, 1 = good).
5. reduced light = (light − bias − dark · t) / master flat, cleaned of cosmic rays (astroscrappy,
   which skips the masked pixels; about 400 per frame), optionally background-subtracted (sep),
   with a simple TAN WCS (reference pixel at the image centre) added to the header

Cosmic-ray removal only works together with the mask: without it, astroscrappy mistakes the
hot pixels for cosmic rays (about 8,700 per frame), removes only their bright part and leaves
dark holes.

"Combined" means the mean per pixel; with sigma clipping on (the default) each pixel is first sigma
clipped along the stack, which rejects cosmic rays in single frames but keeps hot pixels and dust
shadows that are present in every frame. After clipping, the mean gives masters about 15 % less
noisy than the median.

Output is float32: `master_flat_<filter>.fits` and `mask_<filter>.fits` are always written, `master_bias.fits` and
`master_dark.fits` only with `--save-masters` (`SAVE_MASTERS=true` in `postprocess.sh`), and every light frame becomes
`reduced_<original name>`.

## Checking the code

The Python files are type-annotated and checked with [ruff](https://docs.astral.sh/ruff/) (lint
and format) and [ty](https://docs.astral.sh/ty/) (types); both are configured in
`pyproject.toml`:

```sh
uv run ruff check . && uv run ruff format --check . && uv run ty check
```

In addition, every function is decorated with [beartype](https://beartype.readthedocs.io), which
checks the argument and return types on each call while the pipeline runs, including that
frames are float32 arrays.
