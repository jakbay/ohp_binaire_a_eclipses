# import and settings
import numpy as np
import matplotlib.pyplot as plt
import ccdproc as ccdp
import os

from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.coordinates import SkyCoord
from photutils import CircularAperture, aperture_photometry, CircularAnnulus
from photutils import DAOStarFinder
from astropy.stats import mad_std
from scipy.ndimage import shift


## Chargement des donnees
# Fonction pour lister tous les fichiers .fit et .fits dans un dossier donné
def list_fits_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.fit', '.fits'))]

# Dossier contenant les fichiers FITS
folder_path = 'data/obs_2024-09-23/obs_sec' 

# Lister tous les fichiers .fits dans le dossier
fits_files = list_fits_files(folder_path)

print(fits_files)


## Determiner positions des etoiles
# Charger une image FITS
image_data = load_fits_image(fits_file)

# Calculer le bruit de fond (background noise) et la médiane
mean, median, std = np.mean(image_data), np.median(image_data), mad_std(image_data)

# Initialiser DAOStarFinder avec un seuil de détection
daofind = DAOStarFinder(fwhm=40.0, threshold=5.*std)  
sources = daofind(image_data - median)

# Filtrer les colonnes des résultats et afficher
for col in sources.colnames:
    sources[col].info.format = '%.8g'  
print(sources)

# Trier les étoiles par flux décroissant (le flux est proportionnel à la luminosité) -> les 4 premieres etoiles detectees sont les notres
sorted_sources = sources[np.argsort(sources['flux'])[::-1]]

# Extraire les positions des étoiles détectées
star_positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

# Afficher l'image avec les positions des étoiles détectées
plt.figure()
interval = ZScaleInterval()
z1, z2 = interval.get_limits(image_data)
plt.imshow(image_data, origin='lower', vmin=z1, vmax=z2, cmap='gray')

# Tracer les apertures sur les étoiles détectées
apertures = CircularAperture(star_positions, r=80.) 
apertures.plot(color='red', lw=1.5)

plt.title('Étoiles détectées automatiquement')
plt.show()

# Afficher toutes les lignes une par une
for i, source in enumerate(sorted_sources):
    print(f"Étoile {i + 1}: {source}")


## Determiner aperture et annulus
# Positions des étoiles détectées
star_positions = [(1469,1160),(1640, 314), (988, 1848), (392, 1050)]  # Exemple

# Définir l'aperture pour chaque étoile
aperture_radius = 70  # Rayon de l'aperture en pixels
apertures = CircularAperture(star_positions, r=aperture_radius)

# Définir un annulus pour chaque étoile pour estimer le fond
annulus_inner_radius = 130  # Rayon interne de l'annulus
annulus_outer_radius = 200  # Rayon externe de l'annulus
annuli = CircularAnnulus(star_positions, r_in=annulus_inner_radius, r_out=annulus_outer_radius)

# Tracer les apertures et les annuli sur l'image
plt.figure()
plt.imshow(image_data, origin='lower', vmin=z1, vmax=z2, cmap='gray')
apertures.plot(color='red', lw=1.5)
annuli.plot(color='blue', lw=1.5)
plt.title('Apertures et Annuli')
plt.show()


## Courbes d'intensite
# Fonction pour charger une image CCD
def load_fits_image(fits_file):
    hdu = fits.open(fits_file)
    ccd_data = CCDData(hdu[0].data, unit='adu')  
    hdu.close()
    return ccd_data

# Coordonnées de l'étoile cible et des étoiles de référence
star_positions = [(1640, 314), (988, 1848), (392, 1050)] # Coordonnées des étoiles de référence
target_position = (1469,1160)  # Coordonnées de l'étoile cible

# Paramètres de photométrie
aperture_radius = 70  # Rayon de l'aperture pour l'étoile
annulus_inner_radius = 130  # Rayon interne de l'annulus pour le fond
annulus_outer_radius = 200  # Rayon externe de l'annulus pour le fond


# Initialiser des listes pour stocker les résultats d'intensité
target_intensities = []
reference_intensities = []

# Parcourir les fichiers d'observation
for fits_file in fits_files:
    print(f"Traitement du fichier : {fits_file}")
    
    # Charger l'image CCD
    ccd_sci = load_fits_image(fits_file)
    
    # Photométrie pour l'étoile cible
    target_aperture = CircularAperture(target_position, r=aperture_radius)
    target_annulus = CircularAnnulus(target_position, r_in=annulus_inner_radius, r_out=annulus_outer_radius)

    
    # Photométrie pour les étoiles de référence
    reference_apertures = CircularAperture(star_positions, r=aperture_radius)
    reference_annuli = CircularAnnulus(star_positions, r_in=annulus_inner_radius, r_out=annulus_outer_radius)
    
    # Calculer les intensités avec photométrie d'aperture
    phot_table_target = aperture_photometry(ccd_sci.data, target_aperture)
    phot_table_ref = aperture_photometry(ccd_sci.data, reference_apertures)
    
    # Extraire la somme d'intensité pour l'étoile cible et les étoiles de référence
    target_intensity = phot_table_target['aperture_sum'][0]
    reference_intensity = np.mean(phot_table_ref['aperture_sum'])  # Moyenne des intensités des étoiles de référence
    
    # Stocker les intensités
    target_intensities.append(target_intensity)
    reference_intensities.append(reference_intensity)
     
    # Optionnel : Afficher l'image avec les apertures tracées -> ca prend du temps parce que ca va afficher toutes les images mais au moins on pourra check pour voir si toutes les iages sont bien centrees
    plt.figure()
    interval = ZScaleInterval()
    z1, z2 = interval.get_limits(ccd_sci.data)
    plt.imshow(ccd_sci.data, origin='lower', vmin=z1, vmax=z2, cmap='gray')
   
    target_aperture.plot(color='red', lw=2)
    reference_apertures.plot(color='blue', lw=2)
    
    plt.title(f"Image: {fits_file}")
    plt.show()

# Calculer la courbe de lumière normalisée
light_curve = np.array(target_intensities) / np.array(reference_intensities)

# Afficher la courbe de lumière -> ca devrait donner un meilleur resultat une fois les images centrees (cross-fingers)
plt.figure()
plt.plot(light_curve)
plt.xlabel('Observation Number')
plt.ylabel('Normalized Intensity')
plt.title('Light Curve of the Target Star (normalized by reference stars)')
plt.show()
