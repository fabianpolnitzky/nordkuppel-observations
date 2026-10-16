from glob import glob  # for file paths
from os import mkdir, chdir  # for folder creation and switching
from os.path import exists  # for checking if a folder exists

from astropy.io import fits  # for opening the files
from astropy.stats import sigma_clip  # for sigma clipping
from astroscrappy import detect_cosmics  # for cosmics removal

from sep import Background  # for background estimation and calculation

import numpy as np  # for array calculations

# ==================================================================================================================== #
# SETTINGS
# Here the settings for the reduction are specified. The file directory and the names of the specific file types are
# given. Additional settings can be turned on by setting them True, like saving the master correction files or applying
# sigma clipping to the bias and dark files. Cosmics removal for the light files can also be applied with the
# detect_cosmics function from astroscrappy and the background can be subtracted from the files.

file_path = "C:/Users/fabia/Documents/Observations/Nordkuppel/NGC7380/20220719/"  # specify where the files are saved

bias_file = "20220719-*_bias.fit"  # the name of the bias files

dark_file = "20220719-*_dark.fit"  # the name of the dark files

flat_file = "20220719-*_flat_sii.fit"  # the name of the flat files for one filter

light_file = "20220719-*_s2_light.fit"  # the name of the light files for one filter

save_images = (
    False  # saves the master correction files if True (Master Flat is always saved)
)

flat_filter = "s2"  # specifies which filter was used in the saved flat field

sigma_clipping = True  # applies sigma clipping to the dark and flat fields during the reduction process if True

sigma_clipping_value = 3  # the threshold for the values of the sigma clipping

cosmics_removal = False  # applies a cosmics removal algorithm to the light files if True

LaplacianToNoise = 5  # The Laplacian-to-noise limit for cosmic ray detection.

FractionalDetection = 0.3  # The Fractional detection limit for neighboring pixels during the cosmic ray detection.

gain = 1.56  # The gain in electron/ADU for cosmic ray detection.

object_limit = 10.0  # Minimum contrast between Laplacian image and fine structure image.

read_noise = 6.5  # Read noise of the image in electron to generate the noise model of the image

background_subtraction = False  # subtracts the background from the light files

bx, by = 50, 50  # box size along both axis

fx, fy = 3, 3  # window size of 2D median filter along both axis, must both be odd


# ==================================================================================================================== #
# Specification for astrometric solution that is added to the header.
# For each axis the type of the axis and its unit are defined. A reference coordinate point and a point in the image
# as pixels are defined. The parameters for the linear projection matrix are defined.

# First axis

ctype1 = "RA---TAN"  # WCS projection type for this axis

cunit1 = "deg"  # Axis unit

crval1 = 341.8167  # World coordinate on this axis

crpix1 = 2048.00  # Reference pixel of first axis

cd11 = -1.5389e-4  # Linear projection matrix

cd12 = 0.0  # Linear projection matrix

# Second axis

ctype2 = "DEC--TAN"  # WCS projection type for this axis

cunit2 = "deg"  # Axis unit

crval2 = 58.1250  # World coordinate on this axis

crpix2 = 2048.00  # Reference pixel of second axis

cd21 = 0.0  # Linear projection matrix

cd22 = 1.5389e-4  # Linear projection matrix

# ==================================================================================================================== #
# Here the directory is changed to the one where the observational data is saved and a new directory is made if it
# doesn't exist already. The master files and the reduced images will be saved in this folder afterward.

chdir(file_path)

if not exists("Reduced"):
    mkdir("Reduced")

# ==================================================================================================================== #
# MASTER BIAS
# All bias files are added to an array and the master bias is calculated as the mean of all bias files. The master bias
# file can be saved.

path_bias = glob(file_path + bias_file)

biases = np.array([fits.getdata(path) for path in path_bias])

master_bias = np.nanmean(biases, axis=0)

if save_images:
    fits.writeto(
        "Reduced/master_bias.fits", data=master_bias, header=None, overwrite=True
    )

print("Master bias is finished.")

# ==================================================================================================================== #
# MASTER DARK
# It is looped over all dark files. The dark file is read out and the master bias is subtracted. The exposure time from
# the header for the dark file is read out and the dark is divided by it and appended to the list. Sigma clipping can be
# applied to all dark files with the value specified beforehand. The master dark is calculated as the mean and the file
# can be saved.

darks = []

for file in glob(dark_file):
    with fits.open(file) as f:
        dark_b = f[0].data - master_bias

        if sigma_clipping:
            dark_c = sigma_clip(dark_b, sigma=sigma_clipping_value, maxiters=None)
            dark_b = dark_c.filled(np.median(dark_b))

        dark_exp = f[0].header["EXPOSURE"]

        darks.append(dark_b / dark_exp)

master_dark = np.nanmean(darks, axis=0)

if save_images:
    fits.writeto(
        "Reduced/master_dark.fits", data=master_dark, header=None, overwrite=True
    )

print("Master dark is finished.")

# ==================================================================================================================== #
# MASTER FLAT
# It is looped over all flat files. The flat file is read out and the master bias is subtracted. The exposure time from
# the header for the flat file is read out and the master dark is scaled by that value and subtracted from the flat
# file. The flat file is appended to the list and first divided by its mean value. Sigma clipping can be applied to all
# flat files with the value specified beforehand. The master flat is calculated as the mean and the file will always be
# saved for later reduction purpose

flats = []

for file in glob(flat_file):
    with fits.open(file) as f:
        flat_b = f[0].data - master_bias

        flat_exp = f[0].header["EXPOSURE"]
        flat_bd = flat_b - master_dark * flat_exp

        if sigma_clipping:
            flat_c = sigma_clip(flat_bd, sigma=sigma_clipping_value, maxiters=None)
            flat_bd = flat_c.filled(np.median(flat_bd))

        flats.append(flat_bd / np.nanmean(flat_bd))

master_flat = np.nanmean(flats, axis=0)

fits.writeto(
    "Reduced/master_flat_" + flat_filter + ".fits",
    data=master_flat,
    header=None,
    overwrite=True,
)

print("Master flat is finished.")

# ==================================================================================================================== #
# LIGHT REDUCTION
# It is looped over all light files. The light file is read out and the master bias is subtracted. The exposure time
# from the header for the light file is read out and the master dark is scaled by that value and subtracted from the
# light file. The light file is divided by the master flat. If enabled the cosmic rays in the light file are identified
# and removed with the before defined settings. The overall background of the image can be subtracted with the
# SExtractor algorithm with predetermined settings for box and filter sizes. The keywords needed for the astrometric
# solution specified with their value and comment are added to the header. The reduced light file is saved with the
# original header from the light file.

for file in glob(light_file):

    with fits.open(file) as f:

        light_b = f[0].data - master_bias

        light_exp = f[0].header["EXPOSURE"]
        light_bd = light_b - master_dark * light_exp

        light_bdf = light_bd / master_flat

        if cosmics_removal:
            cr_mask, light_bdf = detect_cosmics(
                light_bdf,
                sigclip=LaplacianToNoise,
                sigfrac=FractionalDetection,
                objlim=object_limit,
                gain=gain,
            )
            print("There are {} cosmics detected".format(np.sum(cr_mask)))

        if background_subtraction:
            bkg = Background(light_bdf, bw=bx, bh=by, fw=fx, fh=fy)
            light_bdf = light_bdf - bkg.back()

        f[0].header["CTYPE1"] = (ctype1, "WCS projection type for this axis")

        f[0].header["CUNIT1"] = (cunit1, "Axis unit")

        f[0].header["CRVAL1"] = (crval1, "World coordinate on this axis")

        f[0].header["CRPIX1"] = (crpix1, "Reference pixel of first axis")

        f[0].header["CD1_1"] = (cd11, "Linear projection matrix")

        f[0].header["CD1_2"] = (cd12, "Linear projection matrix")

        f[0].header["CTYPE2"] = (ctype2, "WCS projection type for this axis")

        f[0].header["CUNIT2"] = (cunit2, "Axis unit")

        f[0].header["CRVAL2"] = (crval2, "World coordinate on this axis")

        f[0].header["CRPIX2"] = (crpix2, "Reference pixel of second axis")

        f[0].header["CD2_1"] = (cd21, "Linear projection matrix")

        f[0].header["CD2_2"] = (cd22, "Linear projection matrix")

        fits.writeto(
            "Reduced/reduced_" + file,
            data=light_bdf,
            header=f[0].header,
            overwrite=True,
        )

        print("The reduction of file", file, "is finished.")

print("The whole reduction process is finished.")
