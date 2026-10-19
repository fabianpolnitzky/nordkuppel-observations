"""Bias, dark and flat calibration of Nordkuppel CCD frames for all filters of one night.

Usage::

    python Nordkuppel_Reduction.py NGC7380/20220719
    python Nordkuppel_Reduction.py NGC7380/20220719 --filters ha s2 --output NGC7380/20220719/Reduced_test
    python Nordkuppel_Reduction.py --help

Every setting is a command-line option; the defaults are in ``ReductionSettings``. When run by ``postprocess.sh``,
the values set there are passed on. Output goes to ``<night_dir>/Reduced`` unless ``--output`` is given.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astroscrappy import detect_cosmics  # ty: ignore[unresolved-import]  # compiled, no type information
from beartype import beartype
from cyclopts import App, Parameter
from cyclopts.validators import Number
from numpy.typing import NDArray
from sep import Background  # ty: ignore[unresolved-import]  # compiled, no type information

type Image = NDArray[np.float32]  # one 2D frame

type Mask = NDArray[np.bool_]  # True at bad pixels

type Filter = Literal["ha", "o3", "s2"]  # tag of each filter in the light and output file names

FLAT_TAGS: dict[Filter, str] = {"ha": "ha", "o3": "o3", "s2": "sii"}  # tag of each filter in the flat file names

type Positive = Annotated[float, Parameter(validator=Number(gt=0))]


# ==================================================================================================================== #
# SETTINGS


def odd(type_: object, value: tuple[int, ...]) -> None:
    """Raise a ValueError unless every element of ``value`` is odd; a cyclopts validator."""
    if any(v % 2 == 0 for v in value):
        raise ValueError("must be odd")


@beartype
@dataclass(frozen=True, kw_only=True)
class ReductionSettings:
    """Settings of the reduction, each available as a command-line option.

    Parameters
    ----------
    bias_file : str
        Glob pattern of the bias frames, relative to the night directory.
    dark_file : str
        Glob pattern of the dark frames.
    flat_file : str
        Glob pattern of the flat frames; ``{filter}`` is replaced by the flat tag of each filter (``sii`` for ``s2``).
    light_files : tuple of str
        Glob patterns of the light frames; ``{filter}`` is replaced by the filter. The naming differs between nights.
    save_masters : bool
        Also save the master bias and dark (the master flats are always saved).
    bad_pixel_mask : bool
        Mark hot pixels (from the master dark) and dead or overly sensitive pixels (from the master flat) as bad
        in the weight map ``mask_<filter>.fits`` written next to the reduced frames, so that SExtractor and SWarp
        ignore them. Hot pixels cannot be dark subtracted exactly and otherwise leave specks and rings in the stack.
    hot_sigma : float
        A pixel is hot if its dark current is more than this many standard deviations above the median.
    flat_limits : tuple of float
        A pixel is bad if its normalised flat-field response is outside this range.
    sigma_clipping : bool
        Sigma clip the bias, dark and flat stacks pixel by pixel along the frame axis before combining them.
    clip_sigma : float
        Threshold of the sigma clipping, in standard deviations.
    cosmics_removal : bool
        Remove cosmic rays from the light frames with astroscrappy.
    laplacian_to_noise : float
        Laplacian-to-noise limit for the cosmic-ray detection.
    fractional_detection : float
        Fractional detection limit for neighbouring pixels in the cosmic-ray detection.
    object_limit : float
        Minimum contrast between the Laplacian and the fine-structure image in the cosmic-ray detection.
    gain : float
        Gain (electron/ADU) for the cosmic-ray noise model.
    read_noise : float
        Read noise (electron) for the cosmic-ray noise model; 19 e was measured from bias pairs of 20220712.
    background_subtraction : bool
        Subtract the background of the light frames with sep.
    box_size : tuple of int
        Background box size along x and y (px).
    filter_size : tuple of int
        Size of the median filter of the background map along x and y (boxes); both must be odd.
    wcs_center : tuple of float
        RA and Dec (deg) written as CRVAL1 and CRVAL2, at the image centre; the starting point for SCAMP.
    wcs_cd : tuple of float
        WCS matrix CD1_1, CD1_2, CD2_1, CD2_2 (deg/px). The default is the median SCAMP solution of the NGC 7380
        frames: 0.5543"/px with the x axis mirrored (RA increases with x); the camera rotation differs by about
        2 deg between nights and is left at zero.
    """

    bias_file: str = "*_bias.fit"
    dark_file: str = "*_dark.fit"
    flat_file: str = "*_flat_{filter}.fit"
    light_files: tuple[str, ...] = ("*_light_{filter}.fit", "*_{filter}_light.fit")
    save_masters: bool = False
    bad_pixel_mask: bool = True
    hot_sigma: Positive = 10.0
    flat_limits: tuple[float, float] = (0.5, 1.5)
    sigma_clipping: bool = True
    clip_sigma: Positive = 3.0
    cosmics_removal: bool = True
    laplacian_to_noise: Positive = 5.0
    fractional_detection: Positive = 0.3
    object_limit: Positive = 10.0
    gain: Positive = 1.56
    read_noise: Positive = 19.0
    background_subtraction: bool = False
    box_size: tuple[int, int] = (50, 50)
    filter_size: Annotated[tuple[int, int], Parameter(validator=odd)] = (3, 3)
    wcs_center: tuple[float, float] = (341.7719, 58.0810)
    wcs_cd: tuple[float, float, float, float] = (1.5395e-4, 0.0, 0.0, 1.5395e-4)


# ==================================================================================================================== #
# HELPERS


@beartype
def find_files(night_dir: Path, *patterns: str) -> list[Path]:
    """Return the sorted files in ``night_dir`` matching any of ``patterns``.

    Parameters
    ----------
    night_dir : Path
        Directory of the observing night.
    *patterns : str
        Glob patterns relative to ``night_dir``.

    Returns
    -------
    list of Path
        Matching files, sorted by name.

    Raises
    ------
    FileNotFoundError
        If no file matches, so that an empty stack never produces a NaN master frame.
    """
    files = sorted({path for pattern in patterns for path in night_dir.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No files matching {' or '.join(map(repr, patterns))} in {night_dir}")
    return files


@beartype
def read_frame(path: Path) -> tuple[Image, fits.Header]:
    """Return the primary data of ``path`` as float32 and its header."""
    with fits.open(path) as f:
        return f[0].data.astype(np.float32), f[0].header.copy()


@beartype
def combine(frames: Sequence[Image], clip_sigma: float | None) -> Image:
    """Combine a stack of frames pixel by pixel into one master frame.

    If ``clip_sigma`` is given, each pixel is sigma clipped along the frame axis, so that outliers in single frames
    (cosmic rays, satellite trails) are rejected while structure present in every frame (hot pixels, dust shadows,
    vignetting) is kept. The surviving values are averaged; once the outliers are clipped, the mean gives a less noisy
    master than the median.

    Parameters
    ----------
    frames : list of numpy.ndarray
        Frames of identical shape.
    clip_sigma : float or None
        Clipping threshold in standard deviations, or None for no clipping.

    Returns
    -------
    numpy.ndarray
        The float32 master frame. Pixels rejected in every frame are NaN.
    """
    stack = np.asarray(frames, dtype=np.float32)
    if clip_sigma is not None:
        stack = np.ma.filled(sigma_clip(stack, sigma=clip_sigma, maxiters=None, axis=0), np.nan)
    return np.nanmean(stack, axis=0).astype(np.float32)


@beartype
def save(path: Path, data: Image, header: fits.Header | None = None) -> None:
    """Write ``data`` as float32 to ``path``, overwriting an existing file."""
    fits.writeto(path, data=data.astype(np.float32), header=header, overwrite=True)


# ==================================================================================================================== #
# MASTER FRAMES


@beartype
def make_master_bias(files: Sequence[Path], clip_sigma: float | None) -> Image:
    """Return the master bias as the combined stack of all bias frames, clipped at ``clip_sigma`` if given."""
    return combine([read_frame(path)[0] for path in files], clip_sigma)


@beartype
def make_master_dark(files: Sequence[Path], master_bias: Image, clip_sigma: float | None) -> Image:
    """Return the master dark-current rate.

    Each dark is bias subtracted and divided by its exposure time from the ``EXPOSURE`` header keyword before the stack
    is combined, so the master can be scaled to any exposure time.

    Parameters
    ----------
    files : list of Path
        Dark frames.
    master_bias : numpy.ndarray
        Master bias.
    clip_sigma : float or None
        Clipping threshold for ``combine``, or None for no clipping.

    Returns
    -------
    numpy.ndarray
        Dark current in ADU/s.
    """
    darks = []
    for path in files:
        data, header = read_frame(path)
        darks.append((data - master_bias) / header["EXPOSURE"])
    return combine(darks, clip_sigma)


@beartype
def make_master_flat(files: Sequence[Path], master_bias: Image, master_dark: Image, clip_sigma: float | None) -> Image:
    """Return the normalised master flat for one filter.

    Each flat is bias subtracted, the master dark scaled to its exposure time is subtracted, and the result is divided
    by its median before the stack is combined. The median is only used as the normalisation scale, because it is
    insensitive to hot pixels and stars; any scale common to all flats cancels in the reduction.

    Parameters
    ----------
    files : list of Path
        Flat frames of one filter.
    master_bias : numpy.ndarray
        Master bias.
    master_dark : numpy.ndarray
        Master dark-current rate in ADU/s.
    clip_sigma : float or None
        Clipping threshold for ``combine``, or None for no clipping.

    Returns
    -------
    numpy.ndarray
        Master flat with a median close to one.
    """
    flats = []
    for path in files:
        data, header = read_frame(path)
        flat = data - master_bias - master_dark * header["EXPOSURE"]
        flats.append(flat / np.nanmedian(flat))
    return combine(flats, clip_sigma)


@beartype
def make_bad_pixel_mask(master_dark: Image, master_flat: Image, settings: ReductionSettings) -> Mask:
    """Return the pixels that cannot be calibrated reliably.

    Hot pixels have a dark current far above the rest; their dark subtraction leaves residuals of several sigma with
    either sign, which become specks and, after resampling, rings in the stack. Pixels whose flat-field response is far
    from one, or not finite, are dead or defective.

    Parameters
    ----------
    master_dark : numpy.ndarray
        Master dark-current rate in ADU/s.
    master_flat : numpy.ndarray
        Normalised master flat.
    settings : ReductionSettings
        Source of ``bad_pixel_mask``, ``hot_sigma`` and ``flat_limits``; with ``bad_pixel_mask`` off no pixel is bad.

    Returns
    -------
    numpy.ndarray of bool
        True at bad pixels.
    """
    if not settings.bad_pixel_mask:
        return np.zeros(master_dark.shape, dtype=bool)
    _, median, sigma = sigma_clipped_stats(master_dark)
    hot = master_dark > median + settings.hot_sigma * sigma
    low, high = settings.flat_limits
    with np.errstate(invalid="ignore"):
        defective = ~np.isfinite(master_flat) | (master_flat < low) | (master_flat > high)
    return np.asarray(hot | defective, dtype=bool)


# ==================================================================================================================== #
# LIGHT REDUCTION


@beartype
def add_wcs(header: fits.Header, shape: tuple[int, ...], settings: ReductionSettings) -> None:
    """Write a TAN WCS from ``settings`` into ``header``, with the reference pixel at the image centre.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header to update in place.
    shape : tuple of int
        Image shape as (ny, nx).
    settings : ReductionSettings
        Source of ``wcs_center`` and ``wcs_cd``.
    """
    ny, nx = shape
    cd11, cd12, cd21, cd22 = settings.wcs_cd
    header["CTYPE1"] = ("RA---TAN", "WCS projection type for this axis")
    header["CUNIT1"] = ("deg", "Axis unit")
    header["CRVAL1"] = (settings.wcs_center[0], "World coordinate on this axis")
    header["CRPIX1"] = ((nx + 1) / 2, "Reference pixel of first axis")
    header["CD1_1"] = (cd11, "Linear projection matrix")
    header["CD1_2"] = (cd12, "Linear projection matrix")
    header["CTYPE2"] = ("DEC--TAN", "WCS projection type for this axis")
    header["CUNIT2"] = ("deg", "Axis unit")
    header["CRVAL2"] = (settings.wcs_center[1], "World coordinate on this axis")
    header["CRPIX2"] = ((ny + 1) / 2, "Reference pixel of second axis")
    header["CD2_1"] = (cd21, "Linear projection matrix")
    header["CD2_2"] = (cd22, "Linear projection matrix")


@beartype
def reduce_light(
    path: Path, master_bias: Image, master_dark: Image, master_flat: Image, bad: Mask, settings: ReductionSettings
) -> tuple[Image, fits.Header]:
    """Calibrate one light frame.

    The master bias and the master dark scaled to the exposure time are subtracted and the result is divided by the
    master flat. If enabled, cosmic rays are removed with astroscrappy, which leaves the bad pixels alone (they are
    excluded through the weight map instead), and the background is subtracted with sep. The astrometric solution is
    added to the original header.

    Parameters
    ----------
    path : Path
        Light frame.
    master_bias : numpy.ndarray
        Master bias.
    master_dark : numpy.ndarray
        Master dark-current rate in ADU/s.
    master_flat : numpy.ndarray
        Normalised master flat of the same filter.
    bad : numpy.ndarray of bool
        Bad-pixel mask, passed to astroscrappy.
    settings : ReductionSettings
        Cosmic-ray, background and WCS settings.

    Returns
    -------
    data : numpy.ndarray
        Reduced float32 image.
    header : astropy.io.fits.Header
        Original header with the WCS keywords added.
    """
    data, header = read_frame(path)
    light = (data - master_bias - master_dark * header["EXPOSURE"]) / master_flat

    if settings.cosmics_removal:
        cr_mask, light = detect_cosmics(
            light,
            inmask=bad,
            sigclip=settings.laplacian_to_noise,
            sigfrac=settings.fractional_detection,
            objlim=settings.object_limit,
            gain=settings.gain,
            readnoise=settings.read_noise,
        )
        print(f"There are {np.sum(cr_mask)} cosmics detected")

    if settings.background_subtraction:
        light = np.ascontiguousarray(light, dtype=np.float32)
        (bw, bh), (fw, fh) = settings.box_size, settings.filter_size
        bkg = Background(light, bw=bw, bh=bh, fw=fw, fh=fh)
        light = light - bkg.back()

    add_wcs(header, light.shape, settings)
    return light.astype(np.float32), header


# ==================================================================================================================== #
# MAIN

app = App(help_format="rst", version_flags=(), default_parameter=Parameter(consume_multiple=True, negative_iterable=()))


@app.default
@beartype
def main(
    night_dir: Path,
    *,
    filters: tuple[Filter, ...] = ("ha", "o3", "s2"),
    output: Path | None = None,
    settings: Annotated[ReductionSettings, Parameter(name="*")] = ReductionSettings(),
) -> None:
    """Reduce the light frames of one night for all requested filters.

    Parameters
    ----------
    night_dir : Path
        Directory with the raw frames of one night.
    filters : tuple of str
        Filters to reduce.
    output : Path
        Output directory; default ``<night_dir>/Reduced``.
    """
    out_dir = output if output is not None else night_dir / "Reduced"
    clip_sigma = settings.clip_sigma if settings.sigma_clipping else None

    # Look up every frame type first, so a missing pattern fails before any computation is done.
    bias_files = find_files(night_dir, settings.bias_file)
    dark_files = find_files(night_dir, settings.dark_file)
    filter_files = {
        filt: (
            find_files(night_dir, settings.flat_file.format(filter=FLAT_TAGS[filt])),
            find_files(night_dir, *(pattern.format(filter=filt) for pattern in settings.light_files)),
        )
        for filt in filters
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    master_bias = make_master_bias(bias_files, clip_sigma)
    if settings.save_masters:
        save(out_dir / "master_bias.fits", master_bias)
    print(f"Master bias is finished ({len(bias_files)} frames).")

    master_dark = make_master_dark(dark_files, master_bias, clip_sigma)
    if settings.save_masters:
        save(out_dir / "master_dark.fits", master_dark)
    print(f"Master dark is finished ({len(dark_files)} frames).")

    for filt, (flat_files, light_files) in filter_files.items():
        master_flat = make_master_flat(flat_files, master_bias, master_dark, clip_sigma)
        save(out_dir / f"master_flat_{filt}.fits", master_flat)
        print(f"Master flat {filt} is finished ({len(flat_files)} frames).")

        # Weight map for SExtractor and SWarp: 0 at bad pixels, 1 elsewhere.
        bad = make_bad_pixel_mask(master_dark, master_flat, settings)
        save(out_dir / f"mask_{filt}.fits", (~bad).astype(np.float32))
        print(f"Bad-pixel mask {filt}: {bad.sum()} pixels ({100 * bad.mean():.2f} %).")

        for path in light_files:
            light, header = reduce_light(path, master_bias, master_dark, master_flat, bad, settings)
            save(out_dir / f"reduced_{path.name}", light, header)
        print(f"The reduction of {len(light_files)} {filt} light frames is finished.")

    print("The whole reduction process is finished.")


if __name__ == "__main__":
    app()
