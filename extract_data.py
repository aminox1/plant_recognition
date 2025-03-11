import zipfile
import os

zip_path = "dataplante.zip"  # Fichier ZIP
extract_path = "data/"  # Dossier où extraire les fichiers

# Vérifier si les fichiers sont déjà extraits
if not os.path.exists(extract_path):
    print("Décompression en cours...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extrait avec succès !")
else:
    print("✅ Dataset déjà extrait.")
