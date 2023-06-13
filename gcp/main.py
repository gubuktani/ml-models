from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

potato_model = None
tomato_model = None
apple_model = None
leaf_model = None

BUCKET_NAME = "gubuktani-models"  # Name of the GCP bucket where the models are stored


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def detection(request):
    global potato_model, tomato_model, apple_model, leaf_model

    image = request.files["file"]
    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)))
    image = image / 255
    img_array = tf.expand_dims(image, 0)
    
    class_leaf = ["Daun", "Bukan Daun"]
    
    # Perform leaf detection
    if leaf_model is None:
        download_blob(BUCKET_NAME, "leaf_models/leaf.h5", "/tmp/leaf.h5")
        leaf_model = tf.keras.models.load_model("/tmp/leaf.h5")

    leaf_detection = leaf_model.predict(img_array)
    is_leaf = class_leaf[np.argmax(leaf_detection[0])]

    plant = request.form["plant"]
    if is_leaf == "Daun":
        class_names = []
        detection = []

        if plant == "kentang":
            class_names = ["Busuk awal", "Busuk terlambat", "Sehat"]
            if potato_model is None:
                download_blob(BUCKET_NAME, "potato_models/potatoes.h5", "/tmp/potatoes.h5")
                potato_model = tf.keras.models.load_model("/tmp/potatoes.h5")

            detection = potato_model.predict(img_array)

        elif plant == "tomat":
            class_names = ['Bercak bakteri', 'Busuk awal', 'Busuk terlambat', 'Jamur daun', 'Bercak daun Septoria',
                           'Kutu laba-laba', 'Bercak daun Corynespora', 'Virus keriting daun kuning', 'Virus mozaik', 'Sehat']
            if tomato_model is None:
                download_blob(BUCKET_NAME, "tomato_models/tomatoes.h5", "/tmp/tomatoes.h5")
                tomato_model = tf.keras.models.load_model("/tmp/tomatoes.h5")

            detection = tomato_model.predict(img_array)

        elif plant == "apel":
            class_names = ["Sehat", "Karat", "Scab"]
            if apple_model is None:
                download_blob(BUCKET_NAME, "apple_models/apples.h5", "/tmp/apples.h5")
                apple_model = tf.keras.models.load_model("/tmp/apples.h5")

            detection = apple_model.predict(img_array)

        else:
            return {"status": "error", "message": "Tanaman tidak ditemukan"}

        detected_class = class_names[np.argmax(detection[0])]
        confidence = round(100 * np.max(detection[0]), 2)

        return {"status": "success", "plant": plant, "label": detected_class, "confidence": confidence}
    else:
        return {"status": "error", "message": "Ini bukan gambar daun"}
