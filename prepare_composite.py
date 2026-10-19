"""Crop the per-filter coadds of postprocess.sh to their well-covered area and choose the STIFF levels.

Usage::

    python prepare_composite.py STACK_DIR s2 ha o3 [--crop-shape footprint] [--white-scale 3] [--no-neutral-stars]
    python prepare_composite.py --help

Reads ``coadd_<filter>.fits`` and ``coadd_<filter>.weight.fits`` from STACK_DIR, writes ``crop_<filter>.fits`` there,
and prints one line for STIFF: the comma-separated black points, a space, and the comma-separated white points, both in
the order the filters were given.

The crop keeps the area that reaches a minimum depth in every filter: by default its irregular footprint, with the
rest of the bounding box set below the black point so that it comes out black, or alternatively the largest rectangle
inside it, or the area covered by a minimum number of frames, counted from the frame outlines.

The black point of every filter is its level in the darkest parts of the field (found in the reference filter), so
that dark areas come out neutral. The white point is set by the nebula, not by the stars, so that the faint filters
contribute as much colour as the bright one. Stars are about equally bright in all filters, so this stretch makes them
pink; optionally their light is made neutral inside a star mask (``star_mask.fits``). Every setting is a command-line
option; the defaults are in ``CompositeSettings``, and ``postprocess.sh`` passes on the values set there.
"""

import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
from beartype import beartype
from cyclopts import App, Parameter
from cyclopts.validators import Number
from numpy.typing import NDArray
from sep import Background, extract, mask_ellipse, sum_circle  # ty: ignore[unresolved-import]  # compiled, no types

type Image = NDArray[np.float32]  # one 2D image
type Mask = NDArray[np.bool_]  # True where the pixel is valid
type Box = tuple[int, int, int, int]  # (y0, y1, x0, x1), used as [y0:y1, x0:x1]
type Catalogue = np.ndarray  # structured array of sep detections (fields x, y, a, b, theta, npix, ...)
type CropShape = Literal["footprint", "rectangle", "frames"]
type Fraction = Annotated[float, Parameter(validator=Number(gt=0, lte=1))]
type Positive = Annotated[float, Parameter(validator=Number(gt=0))]


# ==================================================================================================================== #
# SETTINGS


@beartype
@dataclass(frozen=True, kw_only=True)
class CompositeSettings:
    """Settings of the crop, the colour levels and the star colours, each available as a command-line option.

    Parameters
    ----------
    crop_shape : str
        ``frames`` (default) keeps every pixel covered by at least ``min_frames`` frames in every filter, counted
        from the frame outlines (SCAMP solution of each frame), and fills the rest of their bounding box with black,
        so the edge consists only of straight frame edges. ``footprint`` keeps every block that reaches
        ``min_weight`` in all filters instead; its edge follows a depth contour. ``rectangle`` keeps the largest
        rectangle inside that.
    min_frames : int
        Only for ``frames``: minimum number of frames per filter. The noise grows as sqrt(deepest count / count);
        for the two NGC 7380 nights (up to 40 frames per filter) 5 allows at most about 2.8 times the noise of the
        core, 10 at most twice, and 1 keeps the union of all frames.
    single_region : bool
        Only for ``frames``: keep only the connected region that contains the deepest part, dropping detached slivers
        where a few frames happen to overlap at the edge of the field.
    min_weight : float
        Minimum depth: a block is kept if its median weight is at least this fraction of the filter's full-depth
        weight (the 99th percentile of the block medians) in every filter. The noise grows as 1/sqrt(depth), so 0.25
        allows at most twice the noise of the deepest part.
    footprint_smoothing : int
        Only for ``footprint``: passes of a 3 x 3 majority filter over the deep-enough blocks. Where few frames
        overlap, the clipped combine lowers the weight at random, so blocks near ``min_weight`` flip in and out; the
        filter removes those specks and holes but keeps the straight frame edges.
    min_covered : float
        Only for ``rectangle``: the rectangle is shrunk until at least this fraction of it reaches ``min_weight``.
    reference : str
        Filter in which the darkest parts of the field are searched; also the filter against which the colour excess
        of stars is measured.
    block_size : int
        Size (px) of the blocks in which the depth and the darkest parts of the field are measured.
    dark_fraction : float
        Fraction of blocks, the darkest in the reference filter, that define the black point.
    nebula_percentile : float
        Percentile that measures the nebula brightness in each filter.
    white_scale : float
        White point = black point + ``white_scale`` x (nebula brightness - black point); lower is brighter.
    neutral_stars : bool
        Make the light of stars the same in all colour channels, so that they come out white instead of pink. Only
        the light above the local background is changed, so the nebula colour under the stars is kept.
    star_strength : float
        How far the star colours are moved towards neutral: 0 leaves them, 1 makes them fully neutral.
    star_threshold : float
        Detection threshold for stars, in standard deviations of the background noise.
    star_min_excess : float
        A detection counts as a star if, after the stretch, its mean flux in the other filters is at least this many
        times its flux in the reference filter. Stars have about 4, nebula structure about 1 or less.
    star_mask_scale : float
        Size of the mask around each star, in units of the radius of its detected area.
    star_feather : float
        Width (px, Gaussian sigma) of the soft edge of the star mask.
    star_background_box : int
        Box size (px) of the local background under the stars.
    """

    crop_shape: CropShape = "frames"
    min_weight: Fraction = 0.25
    min_frames: Annotated[int, Parameter(validator=Number(gte=1))] = 5
    single_region: bool = True
    footprint_smoothing: Annotated[int, Parameter(validator=Number(gte=0))] = 3
    min_covered: Fraction = 0.99
    reference: str = "ha"
    block_size: Annotated[int, Parameter(validator=Number(gt=0))] = 8
    dark_fraction: Fraction = 0.05
    nebula_percentile: Annotated[float, Parameter(validator=Number(gt=0, lt=100))] = 90.0
    white_scale: Positive = 3.0
    neutral_stars: bool = True
    star_strength: Annotated[float, Parameter(validator=Number(gte=0, lte=1))] = 1.0
    star_threshold: Positive = 3.0
    star_min_excess: Positive = 2.0
    star_mask_scale: Positive = 2.5
    star_feather: Positive = 1.5
    star_background_box: Annotated[int, Parameter(validator=Number(gt=0))] = 32


# ==================================================================================================================== #
# CROP


@beartype
def deep_blocks(weights: Sequence[Image], settings: CompositeSettings) -> Mask:
    """Return the blocks that reach ``min_weight`` of the full depth in every filter.

    The depth is judged on blocks of ``block_size`` pixels by their median weight, because the bad-pixel masks leave
    isolated holes in the coadded weight maps and the resampling adds small overshoots around them.

    Parameters
    ----------
    weights : list of numpy.ndarray
        Weight maps of all filters, on the same pixel grid.
    settings : CompositeSettings
        Source of ``min_weight`` and ``block_size``.

    Returns
    -------
    numpy.ndarray of bool
        One element per block, True where the block is deep enough in all filters.
    """
    block_weights = [np.median(blocks(w, settings.block_size), axis=2) for w in weights]
    return np.asarray(
        np.all([b >= settings.min_weight * np.percentile(b, 99) for b in block_weights], axis=0), dtype=bool
    )


@beartype
def majority(mask: Mask, passes: int) -> Mask:
    """Return ``mask`` after ``passes`` of a 3 x 3 majority filter (each element takes the value of most neighbours)."""
    for _ in range(passes):
        padded = np.pad(mask, 1).astype(np.int8)
        ny, nx = mask.shape
        votes = sum(padded[dy : dy + ny, dx : dx + nx] for dy in range(3) for dx in range(3))
        mask = np.asarray(votes >= 5, dtype=bool)
    return mask


@beartype
def find_rectangle(deep: Mask, min_covered: float) -> tuple[int, int, int, int]:
    """Return the largest box of blocks, found by trimming edges, of which ``min_covered`` is deep enough.

    Starting from the bounding box of all deep blocks, the edge with the smallest deep fraction is moved inwards one
    block at a time until ``min_covered`` of the box is deep enough.

    Parameters
    ----------
    deep : numpy.ndarray of bool
        Deep-enough blocks, from ``deep_blocks``.
    min_covered : float
        Fraction of the box that must be deep enough.

    Returns
    -------
    tuple of int
        ``(y0, y1, x0, x1)`` in blocks.
    """
    ys, xs = np.nonzero(deep)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    while deep[y0:y1, x0:x1].mean() < min_covered:
        edges = {
            "y0": deep[y0, x0:x1].mean(),
            "y1": deep[y1 - 1, x0:x1].mean(),
            "x0": deep[y0:y1, x0].mean(),
            "x1": deep[y0:y1, x1 - 1].mean(),
        }
        worst = min(edges, key=edges.__getitem__)
        if worst == "y0":
            y0 += 1
        elif worst == "y1":
            y1 -= 1
        elif worst == "x0":
            x0 += 1
        else:
            x1 -= 1
    return int(y0), int(y1), int(x0), int(x1)


@beartype
def find_crop(weights: Sequence[Image], settings: CompositeSettings) -> tuple[Box, Mask]:
    """Return the crop box and the mask of valid pixels inside it.

    For ``crop_shape`` ``footprint`` the box is the bounding box of all deep-enough blocks and the mask marks those
    blocks, cleaned by ``majority``, so its edge follows the frame edges in steps of ``block_size``. For
    ``rectangle`` the box is the largest mostly deep rectangle and every pixel in it is valid.

    Parameters
    ----------
    weights : list of numpy.ndarray
        Weight maps of all filters, on the same pixel grid.
    settings : CompositeSettings
        Crop settings.

    Returns
    -------
    box : tuple of int
        ``(y0, y1, x0, x1)`` in pixels, to be used as ``[y0:y1, x0:x1]``.
    valid : numpy.ndarray of bool
        Valid pixels of the box.
    """
    size = settings.block_size
    deep = deep_blocks(weights, settings)
    if settings.crop_shape == "rectangle":
        y0, y1, x0, x1 = find_rectangle(deep, settings.min_covered)
        valid = np.ones(((y1 - y0) * size, (x1 - x0) * size), dtype=bool)
    else:
        deep = majority(deep, settings.footprint_smoothing)
        ys, xs = np.nonzero(deep)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        valid = np.repeat(np.repeat(deep[y0:y1, x0:x1], size, axis=0), size, axis=1)
    return (y0 * size, y1 * size, x0 * size, x1 * size), np.ascontiguousarray(valid)


@beartype
def crop(path: Path, box: Box) -> tuple[Image, fits.Header]:
    """Return the ``box`` of the image at ``path`` as float32 and its header with the WCS updated to the crop."""
    y0, y1, x0, x1 = box
    with fits.open(path) as f:
        cutout = Cutout2D(f[0].data, ((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2), (y1 - y0, x1 - x0), wcs=WCS(f[0].header))
        header = f[0].header.copy()
    if cutout.wcs is None:
        raise ValueError(f"{path} has no WCS")
    header.update(cutout.wcs.to_header())
    return cutout.data.astype(np.float32), header


@beartype
def add_frame(count: NDArray[np.int16], gx: NDArray[np.float64], gy: NDArray[np.float64]) -> None:
    """Add 1 to ``count`` inside the convex quadrilateral with corners ``gx``, ``gy`` (grid pixels), in place."""
    if np.sum(gx * np.roll(gy, -1) - np.roll(gx, -1) * gy) < 0:  # make the corners run anticlockwise
        gx, gy = gx[::-1], gy[::-1]
    ny, nx = count.shape
    x0, x1 = max(int(np.floor(gx.min())), 0), min(int(np.ceil(gx.max())) + 1, nx)
    y0, y1 = max(int(np.floor(gy.min())), 0), min(int(np.ceil(gy.max())) + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = np.ones(xx.shape, dtype=bool)
    for xa, ya, xb, yb in zip(gx, gy, np.roll(gx, -1), np.roll(gy, -1), strict=True):
        inside &= (xb - xa) * (yy - ya) - (yb - ya) * (xx - xa) >= 0
    count[y0:y1, x0:x1] += inside


@beartype
def frame_count(stack_dir: Path, filt: str, grid: WCS, shape: tuple[int, int]) -> NDArray[np.int16]:
    """Return how many frames of ``filt`` cover each pixel of the coadd grid.

    Each frame in ``frames/`` is projected onto the grid through its SCAMP header: its four corners give a
    quadrilateral (the frames are rotated by about 2 deg against each other), which is filled. Frames that SCAMP could
    not match were moved to ``frames/rejected/`` and are not counted.

    Parameters
    ----------
    stack_dir : Path
        Stack directory with ``frames/<name>.head`` and ``frames/<name>.fit``.
    filt : str
        Filter tag in the frame names.
    grid : astropy.wcs.WCS
        WCS of the coadds.
    shape : tuple of int
        Shape (ny, nx) of the coadds.

    Returns
    -------
    numpy.ndarray of int16
        Number of frames per grid pixel.
    """
    count = np.zeros(shape, dtype=np.int16)
    for head in sorted((stack_dir / "frames").glob(f"*_{filt}.head")):
        nx, ny = (fits.getheader(head.with_suffix(".fit"))[key] for key in ("NAXIS1", "NAXIS2"))
        header = fits.Header.fromtextfile(str(head))
        # SCAMP writes TAN with PV distortion terms, which astropy reads correctly as TPV
        header["CTYPE1"], header["CTYPE2"] = "RA---TPV", "DEC--TPV"
        header["NAXIS"], header["NAXIS1"], header["NAXIS2"] = 2, nx, ny
        corner_x = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
        corner_y = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
        ra, dec = WCS(header).pixel_to_world_values(corner_x, corner_y)
        gx, gy = grid.world_to_pixel_values(ra, dec)
        add_frame(count, np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
    return count


@beartype
def connected_region(covered: Mask, seed: tuple[int, int], size: int) -> Mask:
    """Return the pixels of ``covered`` connected to ``seed`` (y, x), traced on blocks of ``size`` pixels for speed."""
    ny, nx = covered.shape
    by, bx = -(-ny // size), -(-nx // size)
    padded = np.zeros((by * size, bx * size), dtype=bool)
    padded[:ny, :nx] = covered
    occupied = padded.reshape(by, size, bx, size).any(axis=(1, 3))
    region = np.zeros_like(occupied)
    region[seed[0] // size, seed[1] // size] = True
    while True:
        grown = np.pad(region, 1)
        grown = np.any([grown[dy : dy + by, dx : dx + bx] for dy in range(3) for dx in range(3)], axis=0) & occupied
        if (grown == region).all():
            break
        region = grown
    return covered & np.repeat(np.repeat(region, size, axis=0), size, axis=1)[:ny, :nx]


@beartype
def find_frames_crop(stack_dir: Path, filters: Sequence[str], settings: CompositeSettings) -> tuple[Box, Mask]:
    """Return the bounding box of all pixels covered by ``min_frames`` frames in every filter, and those pixels.

    Parameters
    ----------
    stack_dir : Path
        Stack directory with the coadds and ``frames/``.
    filters : list of str
        Filters of the colour channels.
    settings : CompositeSettings
        Source of ``min_frames``, ``single_region`` and ``block_size``.

    Returns
    -------
    box : tuple of int
        ``(y0, y1, x0, x1)`` in pixels, to be used as ``[y0:y1, x0:x1]``.
    valid : numpy.ndarray of bool
        Valid pixels of the box.

    Raises
    ------
    ValueError
        If no pixel is covered by ``min_frames`` frames in every filter.
    """
    header = fits.getheader(stack_dir / f"coadd_{filters[0]}.fits")
    grid, shape = WCS(header), (int(header["NAXIS2"]), int(header["NAXIS1"]))
    counts = np.min([frame_count(stack_dir, filt, grid, shape) for filt in filters], axis=0)
    covered = counts >= settings.min_frames
    if not covered.any():
        raise ValueError(f"no pixel is covered by {settings.min_frames} frames in every filter")
    if settings.single_region:
        seed = np.unravel_index(np.argmax(counts), counts.shape)
        covered = connected_region(covered, (int(seed[0]), int(seed[1])), settings.block_size)
    ys, xs = np.nonzero(covered)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    return (y0, y1, x0, x1), np.ascontiguousarray(covered[y0:y1, x0:x1])


# ==================================================================================================================== #
# LEVELS


@beartype
def blocks(data: Image, size: int) -> Image:
    """Return ``data`` cut into ``size`` x ``size`` blocks, shaped (block row, block column, pixels)."""
    ny, nx = (n // size * size for n in data.shape)
    cut = data[:ny, :nx].reshape(ny // size, size, nx // size, size)
    return cut.transpose(0, 2, 1, 3).reshape(ny // size, nx // size, -1)


@beartype
def levels(images: dict[str, Image], valid: Mask, settings: CompositeSettings) -> tuple[list[float], list[float]]:
    """Return the black and white points of every filter for STIFF, measured on the valid pixels only.

    Parameters
    ----------
    images : dict of str to numpy.ndarray
        Cropped coadd of each filter, in the order of the colour channels.
    valid : numpy.ndarray of bool
        Valid pixels of the crop.
    settings : CompositeSettings
        Reference filter, block size, dark fraction, nebula percentile and white scale.

    Returns
    -------
    black : list of float
        Sigma-clipped level of each filter in the darkest valid blocks of the reference filter.
    white : list of float
        ``black + white_scale * (nebula percentile - black)`` of each filter.
    """
    size = settings.block_size
    valid_blocks = np.min(blocks(valid.astype(np.float32), size), axis=2) > 0
    reference_blocks = np.median(blocks(images[settings.reference], size), axis=2)
    dark = valid_blocks & (
        reference_blocks <= np.percentile(reference_blocks[valid_blocks], 100 * settings.dark_fraction)
    )

    black, white = [], []
    for data in images.values():
        _, sky, _ = sigma_clipped_stats(blocks(data, size)[dark])
        nebula = np.percentile(data[valid], settings.nebula_percentile)
        black.append(float(sky))
        white.append(float(sky + settings.white_scale * (nebula - sky)))
    return black, white


# ==================================================================================================================== #
# STAR COLOURS


@beartype
def find_stars(excess: dict[str, Image], invalid: Mask, settings: CompositeSettings) -> tuple[Catalogue, Image]:
    """Return the detections whose colour excess against the reference filter marks them as stars.

    Parameters
    ----------
    excess : dict of str to numpy.ndarray
        Stretched light above the local background in each filter (0 = black point, 1 = white point).
    invalid : numpy.ndarray of bool
        Pixels outside the footprint, ignored by the detection.
    settings : CompositeSettings
        Source of ``reference``, ``star_threshold`` and ``star_min_excess``.

    Returns
    -------
    stars : numpy.ndarray
        sep catalogue of the stars.
    noise : numpy.ndarray
        Local background noise of the detection image (the mean of ``excess``), measured in boxes of
        ``star_background_box``; the depth, and with it the noise, changes across the image.
    """
    box = settings.star_background_box
    detection = np.ascontiguousarray(sum(excess.values()) / len(excess), dtype=np.float32)
    noise = np.ascontiguousarray(
        Background(detection, mask=invalid, bw=box, bh=box, fw=3, fh=3).rms(), dtype=np.float32
    )
    objects = extract(detection, settings.star_threshold, err=noise, mask=invalid, minarea=5)
    radius = 2.5 * np.sqrt(objects["a"] * objects["b"])
    flux = {filt: sum_circle(data, objects["x"], objects["y"], radius)[0] for filt, data in excess.items()}
    others = [flux[filt] for filt in excess if filt != settings.reference]
    colour = np.mean(others, axis=0) / np.maximum(flux[settings.reference], 1e-6)
    return objects[colour >= settings.star_min_excess], noise


@beartype
def star_mask(shape: tuple[int, ...], stars: Catalogue, settings: CompositeSettings) -> Image:
    """Return a soft mask (0 to 1) of ellipses around ``stars``, sized by the radius of each star's detected area."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(stars):
        radius = np.sqrt(stars["npix"] / np.pi) / np.sqrt(stars["a"] * stars["b"])
        mask_ellipse(
            mask, stars["x"], stars["y"], stars["a"], stars["b"], stars["theta"], r=settings.star_mask_scale * radius
        )
    soft = convolve_fft(mask.astype(np.float64), Gaussian2DKernel(settings.star_feather))
    return np.clip(soft, 0, 1).astype(np.float32)


@beartype
def neutralise_stars(
    images: dict[str, Image], valid: Mask, black: Sequence[float], white: Sequence[float], settings: CompositeSettings
) -> tuple[dict[str, Image], Image, int]:
    """Make the light of stars equal in all colour channels after the stretch.

    Each channel is stretched with its black and white point, its local background is removed, and inside the star
    mask the remaining (star) light of each channel is replaced by the mean over the channels, weighted with
    ``star_strength``. The background, and with it the nebula colour under the stars, is not changed. The weight also
    rises from 0 to 1 as the mean star light rises from 0 to ``star_threshold`` times the local noise, so that pixels
    in the mask without real star light (noise, or the dip in the local background around bright stars) keep their
    colour.
    Pixels outside the footprint are ignored.

    Parameters
    ----------
    images : dict of str to numpy.ndarray
        Cropped coadd of each filter, in the order of the colour channels.
    valid : numpy.ndarray of bool
        Valid pixels of the crop.
    black, white : list of float
        Black and white points of the channels, in the same order.
    settings : CompositeSettings
        Star settings.

    Returns
    -------
    images : dict of str to numpy.ndarray
        Coadds with neutral stars, in the units of the input.
    mask : numpy.ndarray
        The soft star mask.
    n_stars : int
        Number of stars found.
    """
    invalid = np.ascontiguousarray(~valid)
    scale = {filt: w - b for filt, b, w in zip(images, black, white, strict=True)}
    offset = dict(zip(images, black, strict=True))
    stretched = {
        filt: np.ascontiguousarray(np.where(valid, (data - offset[filt]) / scale[filt], 0), dtype=np.float32)
        for filt, data in images.items()
    }
    box = settings.star_background_box
    excess = {
        filt: np.where(valid, data - Background(data, mask=invalid, bw=box, bh=box, fw=3, fh=3).back(), 0).astype(
            np.float32
        )
        for filt, data in stretched.items()
    }

    stars, noise = find_stars(excess, invalid, settings)
    mask = star_mask(valid.shape, stars, settings)
    mean = sum(excess.values()) / len(excess)
    signal = np.clip(mean / (settings.star_threshold * np.maximum(noise, 1e-6)), 0, 1)
    weight = settings.star_strength * mask * signal
    neutral = {
        filt: (offset[filt] + (stretched[filt] + weight * (mean - excess[filt])) * scale[filt]).astype(np.float32)
        for filt in images
    }
    return neutral, mask, len(stars)


# ==================================================================================================================== #
# MAIN

app = App(help_format="rst", version_flags=(), default_parameter=Parameter(consume_multiple=True, negative_iterable=()))


@app.default
@beartype
def main(
    stack_dir: Path,
    *filters: str,
    settings: Annotated[CompositeSettings, Parameter(name="*")] = CompositeSettings(),
) -> None:
    """Crop all coadds, optionally make the stars neutral, then print the STIFF black and white points.

    Parameters
    ----------
    stack_dir : Path
        Directory with ``coadd_<filter>.fits`` and their weight maps.
    filters : str
        Filters in the order of the colour channels, e.g. ``s2 ha o3``.
    """
    if settings.reference not in filters:
        sys.exit(f"prepare_composite.py: the reference filter '{settings.reference}' is not among {filters}")
    warnings.simplefilter("ignore", FITSFixedWarning)

    if settings.crop_shape == "frames":
        box, valid = find_frames_crop(stack_dir, filters, settings)
        limit = f">= {settings.min_frames} frames per filter"
    else:
        weights = [fits.getdata(stack_dir / f"coadd_{filt}.weight.fits").astype(np.float32) for filt in filters]
        box, valid = find_crop(weights, settings)
        limit = f"depth >= {settings.min_weight}"
    y0, y1, x0, x1 = box
    print(
        f"Crop ({settings.crop_shape}, {limit}) to x {x0}..{x1}, y {y0}..{y1} "
        f"({x1 - x0} x {y1 - y0} px, {valid.sum() / 1e6:.2f} Mpx with data).",
        file=sys.stderr,
    )

    cropped = {filt: crop(stack_dir / f"coadd_{filt}.fits", box) for filt in filters}
    images = {filt: data for filt, (data, _) in cropped.items()}
    black, white = levels(images, valid, settings)
    for filt, b, w in zip(filters, black, white, strict=True):
        print(f"Levels {filt}: black {b:.5g}, white {w:.5g}", file=sys.stderr)

    if settings.neutral_stars:
        images, mask, n_stars = neutralise_stars(images, valid, black, white, settings)
        fits.writeto(stack_dir / "star_mask.fits", mask, cropped[settings.reference][1], overwrite=True)
        print(
            f"Neutral stars: {n_stars} stars, mask covers {100 * (mask[valid] > 0.5).mean():.1f} % (star_mask.fits).",
            file=sys.stderr,
        )

    # Outside the footprint, one full stretch range below the black point: STIFF renders it black.
    for (filt, data), b, w in zip(images.items(), black, white, strict=True):
        filled = np.where(valid, data, b - (w - b)).astype(np.float32)
        fits.writeto(stack_dir / f"crop_{filt}.fits", filled, cropped[filt][1], overwrite=True)

    print(",".join(f"{b:.6g}" for b in black), ",".join(f"{w:.6g}" for w in white))


if __name__ == "__main__":
    app()
