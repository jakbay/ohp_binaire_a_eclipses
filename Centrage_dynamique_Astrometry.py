import os
import subprocess
import numpy as np
from astropy.io import fits
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from astropy.wcs import WCS


# Répertoire où se trouvent les images et les résultats
image_dir = "data/obs_2024-09-23/obs_sec/"
output_dir = "data/obs_2024-09-23/obs_sec_centered/"

# Liste des fichiers FITS dans le répertoire des images
image_files = [f for f in os.listdir(image_dir) if f.endswith('.fits')]


# Fonction pour résoudre une image avec Astrometry.net
def solve_with_astrometry(image_path, output_dir):
    command = [
        "solve-field",
        "--overwrite",
        "--dir", output_dir,
        "--scale-units", "arcsecperpix",
        "--scale-low", "0.5",  # Ajustez selon la résolution de votre image
        "--scale-high", "1.0",
        image_path
    ]
    subprocess.run(command, check=True)


# Résoudre chaque image avec Astrometry.net
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"Résolution de l'image: {image_file}")
    solve_with_astrometry(image_path, output_dir)


# Lire les fichiers WCS générés
wcs_files = [f for f in os.listdir(output_dir) if f.endswith('.wcs')]


# Fonction pour lire les étoiles résolues par Astrometry.net
def read_astrometry_stars(wcs_file):
    with fits.open(wcs_file) as hdul:
        header = hdul[0].header
    # Extraire les coordonnées WCS du header
    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']
    cd1_1 = header['CD1_1']
    cd1_2 = header['CD1_2']
    cd2_1 = header['CD2_1']
    cd2_2 = header['CD2_2']
    crval1 = header['CRVAL1']
    crval2 = header['CRVAL2']
    return (crpix1, crpix2), (crval1, crval2), [[cd1_1, cd1_2], [cd2_1, cd2_2]]


# Récupérer les informations WCS pour chaque fichier
wcs_data = {}
for wcs_file in wcs_files:
    wcs_path = os.path.join(output_dir, wcs_file)
    print(f"Lecture des informations WCS pour : {wcs_file}")
    wcs_info = read_wcs_info(wcs_path)
    wcs_data[wcs_file] = wcs_info

# Exemple pour afficher les informations WCS du premier fichier
first_wcs_file = wcs_files[0]
print(wcs_data[first_wcs_file])


# Choisissez l'image de référence (par exemple, image1.fits)
ref_wcs_file = wcs_files[0]
ref_wcs = wcs_data[ref_wcs_file]

# Calculer les décalages par rapport à l'image de référence
translations = {}
for wcs_file, current_wcs in wcs_data.items():
    if wcs_file != ref_wcs_file:
        translation = calculate_translation(ref_wcs, current_wcs)
        translations[wcs_file] = translation

# Afficher les translations calculées
for wcs_file, translation in translations.items():
    print(f"Translation pour {wcs_file}: {translation}")


def calculate_translation(ref_wcs, current_wcs):
    """
    Calcule la translation entre les coordonnées WCS de l'image de référence et de l'image actuelle.
    """
    # Extraire les coordonnées célestes des deux images
    ref_crval = ref_wcs[1]
    current_crval = current_wcs[1]
    
    # Calcul de la différence entre les coordonnées célestes
    delta_ra = current_crval[0] - ref_crval[0]  # Différence en RA
    delta_dec = current_crval[1] - ref_crval[1]  # Différence en DEC
    
    # Retourne la translation (décalage) entre les deux images
    return np.array([delta_ra, delta_dec])


def apply_translation(image, translation):
    """
    Applique la translation calculée à une image.
    """
    # Dans un cas réel, ici vous pouvez appliquer la translation aux pixels de l'image
    # Cela peut être effectué par une transformation géométrique, comme une translation
    # ou une transformation affine si nécessaire.
    print(f"Appliquer une translation de {translation} à l'image.")


def model_non_linear_translation(translation_points, target_points):
    """
    Modélise la translation non linéaire entre plusieurs points en utilisant une interpolation spline.
    """
    # Translation points: liste des points de décalage connus
    # Target points: les points cibles où nous voulons appliquer l'interpolation

    # Extraction des coordonnées RA et DEC des points de décalage connus
    known_ra = [point[0] for point in translation_points]
    known_dec = [point[1] for point in translation_points]

    # Création de la fonction d'interpolation spline 2D
    interp_func = interp2d(known_ra, known_dec, kind='cubic')

    # Application de l'interpolation aux points cibles
    interpolated_translations = interp_func(target_points)

    return interpolated_translations


# Supposons que vous avez une liste de points de translation (RA, DEC)
translation_points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]  # Exemple de points de décalage
target_points = [(1.5, 1.5)]  # Exemple de points où on veut interpoler les décalages

# Modéliser les décalages non linéaires
interpolated_translation = model_non_linear_translation(translation_points, target_points)
print(f"Déplacement interpolé pour les points cibles: {interpolated_translation}")


def compensate_translation_to_wcs(wcs_file, translation):
    """
    Applique la compensation de la translation sur les informations WCS et affiche les résultats.
    """
    # Ouvrir le fichier FITS
    with fits.open(wcs_file) as hdul:
        wcs = WCS(hdul[0].header)

    # Appliquer la translation aux coordonnées WCS
    # Ici, nous ajustons simplement les valeurs CRVAL (coordonnées célestes de référence)
    header = hdul[0].header
    header['CRVAL1'] += translation[0]  # RA
    header['CRVAL2'] += translation[1]  # DEC

    # Sauvegarder le nouveau fichier FITS avec la compensation
    hdul.writeto('corrected_' + os.path.basename(wcs_file), overwrite=True)

    print(f"Nouvelle image WCS enregistrée avec les traductions appliquées : corrected_{os.path.basename(wcs_file)}")


# Appliquer la translation aux fichiers WCS
for wcs_file, translation in translations.items():
    wcs_file_path = os.path.join(output_dir, wcs_file)
    compensate_translation_to_wcs(wcs_file_path, translation)
