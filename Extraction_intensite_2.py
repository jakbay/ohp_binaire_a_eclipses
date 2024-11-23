# import and settings
import numpy as np
import matplotlib.pyplot as plt
import ccdproc as ccdp
import os
import pandas as pd

from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord
from photutils import CircularAperture, aperture_photometry, CircularAnnulus
from photutils import DAOStarFinder
from astropy.stats import mad_std
from scipy.ndimage import shift

from scipy.spatial import cKDTree  # Pour faire correspondre les étoiles
from tqdm.notebook import tqdm


# Fonction pour lister tous les fichiers .fit et .fits dans un dossier donné
def list_fits_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.fit', '.fits'))]

# Dossier contenant les fichiers FITS
folder_path = 'data/obs_2024-09-23/obs_sec_bis' 

# Lister tous les fichiers .fits dans le dossier
fits_files = list_fits_files(folder_path)

print(fits_files)


# Fonction pour charger une image FITS
def load_fits_image(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data

# Vérification si des fichiers FITS existent
if not fits_files:
    print(f"Aucun fichier FITS trouvé dans {folder_path}")
    exit()


# Liste pour compiler les intensités des étoiles
intensity_data = []

# Tableau pour stocker les intensités normalisées
normalized_intensities = []

# Nombre d'étoiles à extraire
num_stars_to_extract = 4


# Extraction de l'intensite pour chaque image + Barre d'avancement 
for fits_file in tqdm(fits_files, desc="Traitement des images FITS"):

        # Charger l'image FITS
        image_data = load_fits_image(fits_file)

        # Vérification que l'image n'est pas vide
        if image_data is None:
            print(f"L'image {fits_file} est vide.")
            continue

        # Suppression des valeurs NaN
        image_data = np.nan_to_num(image_data)

        # Calculer le bruit de fond et la médiane
        mean, median, std = np.mean(image_data), np.median(image_data), mad_std(image_data)

        # Initialiser DAOStarFinder avec un seuil de détection
        daofind = DAOStarFinder(fwhm=10.0, threshold=5. * std)
        sources = daofind(image_data - median)

        if sources is None or len(sources) == 0:
            print(f"Aucune étoile détectée dans {fits_file}.")
            intensity_data.append([np.nan] * num_stars_to_extract)
            continue

        # Trier les étoiles par flux décroissant 
        sorted_sources = sources[np.argsort(sources['flux'])[::-1]]

        # Extraire les positions des étoiles détectées
        star_positions = np.transpose((sorted_sources['xcentroid'], sorted_sources['ycentroid']))

        # Afficher l'image avec les positions des étoiles détectées
        plt.figure(figsize=(8, 8))
        interval = ZScaleInterval()
        z1, z2 = interval.get_limits(image_data)
        plt.imshow(image_data, origin='lower', vmin=z1, vmax=z2, cmap='gray')

        # Tracer les apertures sur les étoiles détectées
        apertures = CircularAperture(star_positions, r=80.)
        apertures.plot(color='red', lw=1.5)
        
        # Mesurer le flux des étoiles détectées
        photometry_table = aperture_photometry(image_data, apertures)
        plt.title(f'Étoiles détectées - {os.path.basename(fits_file)}')
        plt.show()

        # Afficher les informations (intensite + position) des 4 premières étoiles détectées
        for i, source in enumerate(sorted_sources[:4]):  # Limiter à 4 étoiles
            print(f"Étoile {i + 1}: Flux = {source['flux']}, Position = ({source['xcentroid']}, {source['ycentroid']})")

        # Ajouter les intensités des 4 premières étoiles dans le tableau ou compléter par des NaN si moins de 4
        intensities = photometry_table['aperture_sum'][:4]  # Limiter à 4 étoiles
        if len(intensities) < num_stars_to_extract:
            intensities = np.pad(intensities, (0, num_stars_to_extract - len(intensities)), constant_values=np.nan)
        intensity_data.append(intensities)

        # Normaliser l'intensité de l'étoile cible par la moyenne des intensités des trois autres
        target_star_intensity = intensities[1]
        ref_stars_mean_intensity = np.mean([intensities[0], intensities[2], intensities[3]])
        normalized_value = target_star_intensity / ref_stars_mean_intensity

        # Ajouter la valeur normalisée à la liste
        normalized_intensities.append(normalized_value)

        print(f"Intensités extraites pour {fits_file}: {intensities}")


# Convertir les données en tableau numpy pour une manipulation facile
intensity_data = np.array(intensity_data)

# Noms des colonnes
column_names = ['Étoile 1', 'RT And', 'Étoile 3', 'Étoile 4']

# Créer un DataFrame à partir des données d'intensité
intensity_df = pd.DataFrame(intensity_data, columns=column_names)

# Afficher le tableau avec les colonnes nommées
print("Tableau des intensités extraites :")
print(intensity_df)

# Tracer l'évolution des intensités pour chaque étoile
plt.figure(figsize=(10, 6))

# Ajouter une courbe pour chaque étoile
for i in range(num_stars_to_extract):
    plt.plot(range(len(fits_files)), intensity_data[:, i], label=f'Étoile {i + 1}')

plt.xlabel('Indice d\'image (ordre temporel)')
plt.ylabel('Intensité')
plt.title('Évolution de l\'intensité des étoiles au cours des observations')
plt.legend()
plt.grid()
plt.show()


# Générer un graphique des intensités normalisées
plt.figure(figsize=(10, 6))
plt.plot(normalized_intensities, marker='o', linestyle='-', color='blue')
plt.xlabel('Numéro de l\'observation')
plt.ylabel('Intensité normalisée')
plt.title('Évolution de l\'intensité normalisée de RT And')
plt.grid()
plt.show()
