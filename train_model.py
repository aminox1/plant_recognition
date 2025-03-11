import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Définition des chemins vers les dossiers du dataset
train_dir = "data/split_ttv_dataset_type_of_plants/Train_Set_Folder/"
val_dir = "data/split_ttv_dataset_type_of_plants/Validation_Set_Folder/"

# Paramètres
img_size = (224, 224)  # Taille des images
batch_size = 32  # Nombre d'images par lot

# Charger les datasets
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)
print("✅ Classes détectées :", train_ds.class_names)

val_ds = keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size
)

# **Stocker les classes avant d'appliquer map()**
class_names = train_ds.class_names
print("✅ Classes détectées :", class_names)

# Normalisation des images (mise à l'échelle 0-1)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Définition du modèle CNN
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),  # **Correction ici**
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")  # Utiliser `class_names`
])

# Compilation du modèle
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Affichage du résumé du modèle
model.summary()

# Entraînement du modèle
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Sauvegarde du modèle
model.save("model/model.h5")
print("✅ Modèle entraîné et sauvegardé !")
