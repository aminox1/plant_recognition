import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Chemins des dossiers
train_dir = "data/split_ttv_dataset_type_of_plants/Train_Set_Folder/"
val_dir = "data/split_ttv_dataset_type_of_plants/Validation_Set_Folder/"

# Charger le dataset d'entraînement
img_size = (224, 224)  # Taille des images
batch_size = 32  # Nombre d'images par lot

train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Afficher les classes détectées
print("✅ Classes détectées :", train_ds.class_names)
# Affichage de quelques images avec leurs labels
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):  # Prendre 1 batch
    for i in range(9):  # Afficher 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # Convertir en image
        plt.title(train_ds.class_names[labels[i]])  # Nom de la classe
        plt.axis("off")

plt.show()