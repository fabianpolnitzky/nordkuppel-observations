#!/usr/bin/env bash
# Full Nordkuppel pipeline: reduction, astrometric calibration, stacking and SHO colour composite.
#
# Usage:
#   ./postprocess.sh OUT_DIR NIGHT_DIR [NIGHT_DIR ...]   full pipeline
#   ./postprocess.sh --render OUT_DIR                    only redo the colour image from the
#                                                        existing stacks in OUT_DIR (about 15 seconds)
# Examples:
#   ./postprocess.sh NGC7380/stack NGC7380/20220712 NGC7380/20220719
#   WHITE_SCALE=2.5 STIFF_GAMMA_FAC=1.2 ./postprocess.sh --render NGC7380/stack
#
# Every setting below can be overridden for a single call by putting VARIABLE=value in front of
# the command, as in the second example; the file itself stays unchanged.
#
# Steps:
#   1. Nordkuppel_Reduction.py (via uv): bias, dark and flat calibration of all filters of every
#      night, written to <NIGHT_DIR>/Reduced
#   2. link all reduced_*.fit frames into OUT_DIR/frames as <date>-<number>_<filter>.fit
#   3. SExtractor: FITS_LDAC source catalogue per frame
#   4. SCAMP: astrometric (Gaia) and relative photometric solution of all frames together,
#      written as one .head file per frame next to the image, where SWarp picks it up
#   5. SWarp: one common output grid for all filters, then one coadd per filter on that grid
#   6. prepare_composite.py (via uv): crop the coadds to the area well covered in every filter,
#      choose per-channel black and white points and, if NEUTRAL_STARS, make the stars white
#   7. STIFF: SHO colour composite (R = SII, G = Halpha, B = OIII) as TIFF
# --render runs only steps 6 and 7.
#
# The astromatic programs are taken from the micromamba environment ASTROMATIC_ENV (see
# environment.yml) if it exists, otherwise from PATH. All their settings are passed as
# -KEYWORD value on the command line, on top of each program's default configuration, so no
# config file needs editing.

set -euo pipefail
shopt -s nullglob

# ============================================================================================== #
# SETTINGS
# Every setting of the pipeline is here; the Python steps receive theirs as command-line options
# (see `uv run python Nordkuppel_Reduction.py --help` and `uv run python prepare_composite.py --help`).
# Booleans are true/false; values with several numbers or patterns are separated by spaces.
# VAR=${VAR:-default} keeps a value given on the command line and uses the default otherwise.

ASTROMATIC_ENV=${ASTROMATIC_ENV:-astromatic}  # micromamba environment with the astromatic programs
FILTERS=${FILTERS:-"ha o3 s2"}                # filter tags as used in the light and reduced file names
GAIN=${GAIN:-1.56}                            # electron/ADU, for the cosmic-ray noise model and SExtractor

# Reduction (Nordkuppel_Reduction.py); {filter} in a pattern is replaced by the filter tag
BIAS_FILE=${BIAS_FILE:-"*_bias.fit"}          # glob pattern of the bias frames in a night directory
DARK_FILE=${DARK_FILE:-"*_dark.fit"}          # glob pattern of the dark frames
FLAT_FILE=${FLAT_FILE:-"*_flat_{filter}.fit"} # glob pattern of the flat frames (the flat tag of s2 is sii)
LIGHT_FILES=${LIGHT_FILES:-"*_light_{filter}.fit *_{filter}_light.fit"}  # light frames; naming differs per night
SAVE_MASTERS=${SAVE_MASTERS:-false}           # also save the master bias and dark
BAD_PIXEL_MASK=${BAD_PIXEL_MASK:-true}        # give hot and dead pixels weight 0 (mask_<filter>.fits)
HOT_SIGMA=${HOT_SIGMA:-10.0}                  # hot pixel: dark current above median + HOT_SIGMA sigma
FLAT_LIMITS=${FLAT_LIMITS:-"0.5 1.5"}         # dead/defective pixel: flat response outside this range
SIGMA_CLIPPING=${SIGMA_CLIPPING:-true}        # sigma clip the calibration stacks pixel by pixel
CLIP_SIGMA=${CLIP_SIGMA:-3.0}                 # threshold of that clipping (standard deviations)
COSMICS_REMOVAL=${COSMICS_REMOVAL:-true}      # remove cosmic rays from the light frames (astroscrappy)
LAPLACIAN_TO_NOISE=${LAPLACIAN_TO_NOISE:-5.0} # cosmic rays: Laplacian-to-noise limit
FRACTIONAL_DETECTION=${FRACTIONAL_DETECTION:-0.3}  # cosmic rays: fractional detection limit for neighbours
OBJECT_LIMIT=${OBJECT_LIMIT:-10.0}            # cosmic rays: contrast limit against fine structure
READ_NOISE=${READ_NOISE:-19.0}                # cosmic rays: read noise (electron), measured from bias pairs
BACKGROUND_SUBTRACTION=${BACKGROUND_SUBTRACTION:-false}  # subtract the background of each light frame (sep)
BOX_SIZE=${BOX_SIZE:-"50 50"}                 # background box size along x and y (px)
FILTER_SIZE=${FILTER_SIZE:-"3 3"}             # background median filter along x and y (boxes, odd)
WCS_CENTER=${WCS_CENTER:-"341.7719 58.0810"}  # RA and Dec (deg) of the starting WCS for SCAMP
WCS_CD=${WCS_CD:-"1.5395e-4 0.0 0.0 1.5395e-4"}  # CD1_1 CD1_2 CD2_1 CD2_2 (deg/px) of the starting WCS

# Source extraction, astrometry and stacking
DETECT_THRESH=${DETECT_THRESH:-1.5}           # SExtractor detection and analysis threshold (sigma)
ASTREF_CATALOG=${ASTREF_CATALOG:-GAIA-EDR3}   # SCAMP reference catalogue (downloaded from VizieR)
ASTREFMAG_LIMITS=${ASTREFMAG_LIMITS:-"-5.0,15.0"}  # magnitude range of reference stars
POSITION_MAXERR=${POSITION_MAXERR:-30.0}      # SCAMP search radius (arcmin) around the header position;
                                              # the pointing wanders by up to ~20' from WCS_CENTER
MIN_XY_CONTRAST=${MIN_XY_CONTRAST:-3}         # frames whose SCAMP match is weaker than this are left out
SKY_BACK_SIZE=${SKY_BACK_SIZE:-2048}          # SWarp background mesh (px); >= image size fits one constant
                                              # level per frame, so the extended nebula is not subtracted
COMBINE_TYPE=${COMBINE_TYPE:-CLIPPED}         # SWarp combination; CLIPPED rejects cosmics and satellites

# Colour composite (prepare_composite.py and STIFF); these are all that --render uses
RGB_FILTERS=${RGB_FILTERS:-"s2 ha o3"}        # filters for red, green and blue (SHO palette)
CROP_SHAPE=${CROP_SHAPE:-frames}              # crop: frames (area of >= MIN_FRAMES frames, black outside: only
                                              # straight frame edges), footprint (depth contour) or rectangle
MIN_WEIGHT=${MIN_WEIGHT:-0.25}                # crop: minimum depth in every filter, as fraction of full depth;
                                              # noise grows as 1/sqrt(depth), so 0.25 = at most 2x the noise
MIN_COVERED=${MIN_COVERED:-0.99}              # crop, rectangle only: fraction of it that must reach MIN_WEIGHT
FOOTPRINT_SMOOTHING=${FOOTPRINT_SMOOTHING:-3}  # crop, footprint only: majority-filter passes that clean its edge
MIN_FRAMES=${MIN_FRAMES:-5}                   # crop, frames only: frames per filter; for both NGC 7380 nights
                                              # 5 = at most ~2.8x the noise, 10 = 2x, 1 = union of all frames
SINGLE_REGION=${SINGLE_REGION:-true}          # crop, frames only: drop detached slivers, keep the core region
DARK_REFERENCE=${DARK_REFERENCE:-ha}          # levels: filter in which the darkest parts of the field are found
BLOCK_SIZE=${BLOCK_SIZE:-8}                   # crop and levels: block size (px) for coverage and darkest parts
DARK_FRACTION=${DARK_FRACTION:-0.05}          # levels: fraction of darkest blocks that sets the black point
NEBULA_PERCENTILE=${NEBULA_PERCENTILE:-90}    # levels: percentile that measures the nebula brightness
WHITE_SCALE=${WHITE_SCALE:-3.0}               # white point per channel, in units of its nebula brightness
                                              # above the black point; lower is brighter
# Optional colour steps towards the gold-and-blue SHO look; each is switched on or off on its own
SCALE_CHANNELS=${SCALE_CHANNELS:-true}        # multiply each channel's brightness by CHANNEL_SCALE
CHANNEL_SCALE=${CHANNEL_SCALE:-"1.15 1.0 0.85"}  # factors for red, green, blue: a bit more SII and less OIII
                                              # shifts the balance from blue towards gold; a green factor
                                              # below 1 tints faint areas magenta
RAISE_BLACK=${RAISE_BLACK:-true}              # raise the black point above the faint nebula and its noise
BLACK_OFFSET=${BLACK_OFFSET:-0.04}            # the raise, as a fraction of each channel's stretch range
SMOOTH_COLOUR=${SMOOTH_COLOUR:-true}          # smooth the noisy SII and OIII (all but DARK_REFERENCE), not the stars
COLOUR_SIGMA=${COLOUR_SIGMA:-1.5}             # width of that smoothing (px, Gaussian sigma)
SMOOTH_FAINT=${SMOOTH_FAINT:-true}            # smooth DARK_REFERENCE (Halpha) in its faint parts only
FAINT_SIGMA=${FAINT_SIGMA:-1.5}               # width of that smoothing (px, Gaussian sigma)
FAINT_LEVEL=${FAINT_LEVEL:-0.15}              # sharp from this fraction of its stretch range up
SUPPRESS_GREEN=${SUPPRESS_GREEN:-true}        # lower green where it exceeds the mean of red and blue after the
                                              # stretch: orange Halpha/SII, blue OIII instead of all green
GREEN_SUPPRESSION=${GREEN_SUPPRESSION:-1.0}   # fraction of that excess removed (1: green never exceeds the mean)
SMOOTH_SUPPRESSION=${SMOOTH_SUPPRESSION:-true}  # measure the excess on smoothed copies (COLOUR_SIGMA); per pixel,
                                              # the noise turns faint areas magenta
NEUTRAL_STARS=${NEUTRAL_STARS:-true}          # make star light neutral (white) instead of pink; the
                                              # nebula colour under the stars is kept
STAR_STRENGTH=${STAR_STRENGTH:-1.0}           # stars: 0 keeps their colour, 1 makes them fully neutral
STAR_THRESHOLD=${STAR_THRESHOLD:-3.0}         # stars: detection threshold (sigma)
STAR_MIN_EXCESS=${STAR_MIN_EXCESS:-2.0}       # stars: min. stretched flux in the other filters vs. DARK_REFERENCE
STAR_MASK_SCALE=${STAR_MASK_SCALE:-2.5}       # stars: mask size in units of each star's detected radius
STAR_FEATHER=${STAR_FEATHER:-1.5}             # stars: soft mask edge (px)
STAR_BACKGROUND_BOX=${STAR_BACKGROUND_BOX:-32}  # stars: box (px) for the local background under them
STIFF_GAMMA=${STIFF_GAMMA:-2.2}               # STIFF display gamma (power-law stretch)
STIFF_GAMMA_FAC=${STIFF_GAMMA_FAC:-1.0}       # STIFF stretch; above 1 brightens the faint nebula
STIFF_COLOUR_SAT=${STIFF_COLOUR_SAT:-1.3}     # STIFF colour saturation (1 unchanged)
STIFF_BITS=${STIFF_BITS:-16}                  # bits per colour channel of the TIFF (8 or 16)
RGB_NAME=${RGB_NAME:-SHO.tif}                 # name of the colour composite

# ============================================================================================== #
# HELPERS

SCRIPT=$(basename "$0")             # name used in messages, so they follow a rename of this file

die() { echo "$SCRIPT: $*" >&2; exit 1; }

usage() {
    die "usage: $0 OUT_DIR NIGHT_DIR [NIGHT_DIR ...]   or   $0 --render OUT_DIR"
}

# Print the first of the given program names that is on PATH.
find_tool() {
    local name
    for name in "$@"; do
        if command -v "$name" > /dev/null; then echo "$name"; return; fi
    done
    die "none of '$*' found on PATH"
}

# Turn the header of a FITS file into an ASCII .head file (one 80-character card per line).
fits_to_head() {
    fold -w 80 "$1" | awk '{ print } /^END( |$)/ { exit }' > "$2"
}

# ============================================================================================== #
# STEP 1: REDUCTION

reduce_nights() {
    local night light_files
    # read -a splits without expanding the glob patterns
    read -r -a light_files <<< "$LIGHT_FILES"
    # shellcheck disable=SC2086  # the word lists (FILTERS, BOX_SIZE, ...) are split on purpose
    local options=(
        --filters $FILTERS
        --bias-file "$BIAS_FILE"
        --dark-file "$DARK_FILE"
        --flat-file "$FLAT_FILE"
        --light-files "${light_files[@]}"
        --save-masters="$SAVE_MASTERS"
        --bad-pixel-mask="$BAD_PIXEL_MASK"
        --hot-sigma "$HOT_SIGMA"
        --flat-limits $FLAT_LIMITS
        --sigma-clipping="$SIGMA_CLIPPING"
        --clip-sigma "$CLIP_SIGMA"
        --cosmics-removal="$COSMICS_REMOVAL"
        --laplacian-to-noise "$LAPLACIAN_TO_NOISE"
        --fractional-detection "$FRACTIONAL_DETECTION"
        --object-limit "$OBJECT_LIMIT"
        --gain "$GAIN"
        --read-noise "$READ_NOISE"
        --background-subtraction="$BACKGROUND_SUBTRACTION"
        --box-size $BOX_SIZE
        --filter-size $FILTER_SIZE
        --wcs-center $WCS_CENTER
        --wcs-cd $WCS_CD
    )
    for night in "${nights[@]}"; do
        echo "Reducing $night"
        "$UV" run --project "$REPO" python "$REPO/Nordkuppel_Reduction.py" "$night" "${options[@]}"
    done
}

# ============================================================================================== #
# STEP 2: INPUT FRAMES

link_frames() {
    local night dir files f name id filt candidate
    # frames/ holds only links, catalogues and headers made here; clearing it keeps frames of
    # nights from an earlier run with the same OUT_DIR out of this stack
    rm -rf "$OUT/frames"
    mkdir -p "$OUT/frames"
    for night in "${nights[@]}"; do
        dir=$night/Reduced
        files=("$dir"/reduced_*.fit)
        [ ${#files[@]} -gt 0 ] || die "no reduced_*.fit in $dir"
        for f in "${files[@]}"; do
            name=$(basename "$f" .fit)
            name=${name#reduced_}
            id=${name%%_*}
            filt=""
            for candidate in $FILTERS; do
                case "$name" in
                    *_light_"$candidate" | *_"$candidate"_light) filt=$candidate ;;
                esac
            done
            [ -n "$filt" ] || die "cannot tell the filter of $f"
            [ -f "$dir/mask_$filt.fits" ] || die "no weight map $dir/mask_$filt.fits"
            ln -sf "$f" "$OUT/frames/${id}_${filt}.fit"
            # SWarp finds X.weight.fits next to X.fit by itself (WEIGHT_SUFFIX)
            ln -sf "$dir/mask_$filt.fits" "$OUT/frames/${id}_${filt}.weight.fits"
        done
    done
    frames=(frames/*.fit)
    echo "Linked ${#frames[@]} frames into $OUT/frames"
}

# ============================================================================================== #
# STEP 3: SEXTRACTOR

extract_sources() {
    local img others
    "$SEX" -d > sextractor.config

    # The conda-forge build warns about single-threaded ATLAS on every call; that only matters for
    # model fitting, which is not used here. All messages go to a log that is shown in full if
    # SExtractor fails; on success every warning except the ATLAS one is still printed below.
    : > sextractor.log
    for img in "${frames[@]}"; do
        "$SEX" "$img" -c sextractor.config \
            -CATALOG_NAME "${img%.fit}.cat" \
            -CATALOG_TYPE FITS_LDAC \
            -PARAMETERS_NAME "$REPO/sextractor.param" \
            -FILTER_NAME "$REPO/gauss_2.5_5x5.conv" \
            -DETECT_THRESH "$DETECT_THRESH" \
            -ANALYSIS_THRESH "$DETECT_THRESH" \
            -GAIN "$GAIN" \
            -WEIGHT_TYPE MAP_WEIGHT \
            -WEIGHT_IMAGE "${img%.fit}.weight.fits" \
            -VERBOSE_TYPE QUIET \
            >> sextractor.log 2>&1 \
            || { cat sextractor.log >&2; die "SExtractor failed on $img"; }
    done
    others=$(tr -d '\033' < sextractor.log | grep -i 'warning' | grep -v 'ATLAS library' | sort | uniq -c || true)
    if [ -n "$others" ]; then
        echo "$SCRIPT: SExtractor warnings (number of frames, message):" >&2
        echo "$others" >&2
    fi
    echo "SExtractor catalogues finished (messages in sextractor.log)."
}

# ============================================================================================== #
# STEP 4: SCAMP

solve_astrometry() {
    local bad name contrast
    "$SCAMP" -d > scamp.config

    # The conda-forge SCAMP is built without PLPlot, so it cannot draw check plots.
    "$SCAMP" frames/*.cat -c scamp.config \
        -ASTREF_CATALOG "$ASTREF_CATALOG" \
        -ASTREFMAG_LIMITS "$ASTREFMAG_LIMITS" \
        -PIXSCALE_MAXERR 1.2 \
        -POSANGLE_MAXERR 180.0 \
        -POSITION_MAXERR "$POSITION_MAXERR" \
        -MATCH_FLIPPED Y \
        -CHECKPLOT_TYPE NONE \
        -XML_NAME scamp.xml
    echo "SCAMP solution finished."

    # A frame whose pattern matching failed gets a wrong position and would smear the stack, so it
    # is moved out of the way. Successful matches have an XY contrast well above 10.
    bad=$("$UV" run --project "$REPO" python - scamp.xml "$MIN_XY_CONTRAST" << 'EOF'
import sys, warnings
from astropy.io.votable import parse
from astropy.io.votable.exceptions import VOWarning
warnings.simplefilter("ignore", VOWarning)  # SCAMP's XML may deviate from the VOTable standard
fields = next(t for t in parse(sys.argv[1]).iter_tables() if t.name == "Fields").array
for name, contrast in zip(fields["Catalog_Name"], fields["XY_Contrast"]):
    if contrast < float(sys.argv[2]):
        print(name.removesuffix(".cat"), round(float(contrast), 1))
EOF
)
    if [ -n "$bad" ]; then
        mkdir -p frames/rejected
        while read -r name contrast; do
            echo "$SCRIPT: WARNING: SCAMP could not match $name (XY contrast $contrast), left out" >&2
            mv "frames/$name".* frames/rejected/
        done <<< "$bad"
    fi
}

# ============================================================================================== #
# STEP 5: SWARP

coadd() {
    local filt
    "$SWARP" -d > swarp.config

    # A header-only run over all frames defines one output grid, which every filter is resampled
    # to. SWarp reads <IMAGEOUT_NAME without .fits>.head as the target header.
    "$SWARP" frames/*.fit -c swarp.config \
        -HEADER_ONLY Y \
        -IMAGEOUT_NAME grid.fits \
        -WEIGHTOUT_NAME grid.weight.fits \
        -VERBOSE_TYPE QUIET
    fits_to_head grid.fits grid.head
    rm -f grid.fits grid.weight.fits

    # The weight maps are the bad-pixel masks (0 or 1); RESCALE_WEIGHTS scales each to the noise SWarp
    # measures in its frame, so frames are still weighted by their noise and bad pixels are ignored.
    mkdir -p resamp
    for filt in $FILTERS; do
        cp grid.head "coadd_$filt.head"
        "$SWARP" frames/*_"$filt".fit -c swarp.config \
            -IMAGEOUT_NAME "coadd_$filt.fits" \
            -WEIGHTOUT_NAME "coadd_$filt.weight.fits" \
            -WEIGHT_TYPE MAP_WEIGHT \
            -RESCALE_WEIGHTS Y \
            -COMBINE_TYPE "$COMBINE_TYPE" \
            -SUBTRACT_BACK Y \
            -BACK_SIZE "$SKY_BACK_SIZE" \
            -RESAMPLE_DIR resamp \
            -XML_NAME "swarp_$filt.xml"
        echo "Coadd of filter $filt finished."
    done
}

# ============================================================================================== #
# STEPS 6 AND 7: CROP, LEVELS AND STIFF

render() {
    local levels black white filt gamma crops=()
    echo "Rendering with WHITE_SCALE=$WHITE_SCALE STIFF_GAMMA_FAC=$STIFF_GAMMA_FAC" \
        "STIFF_COLOUR_SAT=$STIFF_COLOUR_SAT NEUTRAL_STARS=$NEUTRAL_STARS SCALE_CHANNELS=$SCALE_CHANNELS" \
        "RAISE_BLACK=$RAISE_BLACK SMOOTH_COLOUR=$SMOOTH_COLOUR SMOOTH_FAINT=$SMOOTH_FAINT" \
        "SUPPRESS_GREEN=$SUPPRESS_GREEN"
    # the green suppression is measured in STIFF's stretch, x^(1 / (GAMMA * GAMMA_FAC))
    gamma=$(awk -v g="$STIFF_GAMMA" -v f="$STIFF_GAMMA_FAC" 'BEGIN { print g * f }')

    # The pointings wander by up to a frame width, so only the centre is well covered in every
    # filter. The black point of each channel is its level in the darkest parts of the field and
    # the white point is set by the nebula, not the stars; prints "black1,black2,black3 white1,...".
    # shellcheck disable=SC2086  # RGB_FILTERS and CHANNEL_SCALE are word lists on purpose
    levels=$("$UV" run --project "$REPO" python "$REPO/prepare_composite.py" . $RGB_FILTERS \
        --crop-shape "$CROP_SHAPE" \
        --min-weight "$MIN_WEIGHT" \
        --min-covered "$MIN_COVERED" \
        --footprint-smoothing "$FOOTPRINT_SMOOTHING" \
        --min-frames "$MIN_FRAMES" \
        --single-region="$SINGLE_REGION" \
        --reference "$DARK_REFERENCE" \
        --block-size "$BLOCK_SIZE" \
        --dark-fraction "$DARK_FRACTION" \
        --nebula-percentile "$NEBULA_PERCENTILE" \
        --white-scale "$WHITE_SCALE" \
        --scale-channels="$SCALE_CHANNELS" \
        --channel-scale $CHANNEL_SCALE \
        --raise-black="$RAISE_BLACK" \
        --black-offset "$BLACK_OFFSET" \
        --smooth-colour="$SMOOTH_COLOUR" \
        --colour-sigma "$COLOUR_SIGMA" \
        --smooth-faint="$SMOOTH_FAINT" \
        --faint-sigma "$FAINT_SIGMA" \
        --faint-level "$FAINT_LEVEL" \
        --suppress-green="$SUPPRESS_GREEN" \
        --green-suppression "$GREEN_SUPPRESSION" \
        --smooth-suppression="$SMOOTH_SUPPRESSION" \
        --gamma "$gamma" \
        --neutral-stars="$NEUTRAL_STARS" \
        --star-strength "$STAR_STRENGTH" \
        --star-threshold "$STAR_THRESHOLD" \
        --star-min-excess "$STAR_MIN_EXCESS" \
        --star-mask-scale "$STAR_MASK_SCALE" \
        --star-feather "$STAR_FEATHER" \
        --star-background-box "$STAR_BACKGROUND_BOX")
    black=${levels% *}
    white=${levels#* }

    "$STIFF" -d > stiff.config
    for filt in $RGB_FILTERS; do crops+=("crop_$filt.fits"); done

    "$STIFF" "${crops[@]}" -c stiff.config \
        -OUTFILE_NAME "$RGB_NAME" \
        -BITS_PER_CHANNEL "$STIFF_BITS" \
        -MIN_TYPE MANUAL -MIN_LEVEL "$black" \
        -MAX_TYPE MANUAL -MAX_LEVEL "$white" \
        -GAMMA "$STIFF_GAMMA" \
        -GAMMA_FAC "$STIFF_GAMMA_FAC" \
        -COLOUR_SAT "$STIFF_COLOUR_SAT" \
        -XML_NAME stiff.xml
    echo "Colour composite written to $OUT/$RGB_NAME"
}

# ============================================================================================== #
# MAIN

render_only=false
if [ "${1:-}" = --render ]; then
    render_only=true
    shift
    [ $# -eq 1 ] || usage
else
    [ $# -ge 2 ] || usage
fi

REPO=$(cd "$(dirname "$0")" && pwd)
OUT=$1; shift

if command -v micromamba > /dev/null; then
    set +u   # the activation scripts use unset variables
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "$ASTROMATIC_ENV" \
        || echo "$SCRIPT: cannot activate micromamba env '$ASTROMATIC_ENV', using PATH" >&2
    set -u
fi

# Look up every program first, so a missing one fails before anything has run.
UV=$(find_tool uv)
STIFF=$(find_tool stiff STIFF)

if $render_only; then
    [ -d "$OUT" ] || die "no such stack directory: $OUT"
    OUT=$(cd "$OUT" && pwd)
    cd "$OUT"
    for filt in $RGB_FILTERS; do
        [ -f "coadd_$filt.fits" ] && [ -f "coadd_$filt.weight.fits" ] \
            || die "no coadd_$filt.fits and weight map in $OUT; run the full pipeline first"
    done
else
    SEX=$(find_tool source-extractor sex)
    SCAMP=$(find_tool scamp)
    SWARP=$(find_tool SWarp swarp)

    nights=()
    for night in "$@"; do
        [ -d "$night" ] || die "no such night directory: $night"
        nights+=("$(cd "$night" && pwd)")
    done

    reduce_nights
    mkdir -p "$OUT"
    OUT=$(cd "$OUT" && pwd)
    cd "$OUT"
    link_frames
    extract_sources
    solve_astrometry
    coadd
fi

render
