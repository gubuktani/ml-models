<h1 align="center">
  <img align="center" src="images\logo.jpg"  width="250"></img>
<br>
GUBUK TANI MACHINE LEARNING PROJECT README
</h1>

# Profile

### Team ID : C23-PR542

## Members

* (ML) M166DKX3804 - Arya Bramaputra (Diponegoro University)
* (ML) M169DSX0467 - Saeful Ismail (Gadjah Mada University)
* (ML) M158DSX3169 - Wickly Gusthvi (Universitas Cenderawasih)
* (CC) C136DSX3008 - Sahid Anwar (Amikom University)
* (CC) C136DSY2778 - Galih Kusuma Dewi (Amikom University)
* (MD) A136DKX4503 - Mico Yumna Ardhana (Amikom University)

# Gubuk Tani Machine Learning Project

This ML project is our final project for Bangkit Academy 2023 Batch 1.

### Android: [Gubuk Tani Android App](https://github.com/gubuktani/md-gubuk-tani-app)

### Cloud: [Gubuk Tani Backend](https://github.com/gubuktani/cc-backend-api) & [Gubuk Tani Frontend](https://github.com/gubuktani/cc-frontend-cms)

## **Project Background:**

**Machine Learning:**

Building four kinds of models that include [Apple leaf disease](https://github.com/gubuktani/MachineLearning-GubukTani/blob/main/Notebook/Apples.ipynb), [Tomato leaf disease](https://github.com/gubuktani/MachineLearning-GubukTani/blob/main/Notebook/Tomato_leaf_disease.ipynb), [Potato leaf disease](https://github.com/gubuktani/MachineLearning-GubukTani/blob/main/Notebook/Gubuktani_Potato_disease.ipynb), [Rice leaf disease](https://github.com/gubuktani/ml-models/blob/main/Notebook/Rice_disease.ipynb) and [Leaf detection](https://github.com/gubuktani/MachineLearning-GubukTani/blob/main/Notebook/leafDetection.ipynb). Build process using * CNN and Pre-trained model or transfer learning by *resnet50v2, nasnetmobile, inceptionv3, mobilenetv2, vgg16*. The model was saved with *model.h5* and chosen by the [best model](https://github.com/gubuktani/MachineLearning-GubukTani/tree/main/model) for deployment.

**Project Case :**

- Apple Leaf Diseases
- Tomato Leaf Diseases
- Potato Leaf Diseases
- Rice Leaf Disease

**Dataset Link:**

- Apple Diseases
  - [Apple Leaf Diseases Datasets](https://drive.google.com/drive/folders/1ecSphBr8TIXYt4OsOa6zVEPyHyAiRn2p?usp=sharing)
- Tomato Leaf Diseases

  - [Tomato Leaf Disease Datasets](https://www.kaggle.com/datasets/noulam/tomato)

- Potato Plant Diseases

  - [Potato Leaf Disease Datasets](https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset)

- Rice Plant Disease
  - [Rice Plant Disease Datasets](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)

- Leaf Detection
  - [Leaf Datasets](https://www.kaggle.com/datasets/fabinahian/plant-disease-45-classes)
  - [Not Leaf Datasets](https://www.kaggle.com/datasets/lijiyu/imagenet)

## Notebook for each case

### Apple Disease

- [Apple leaf disease](https://github.com/gubuktani/MachineLearning-GubukTani/blob/main/Notebook/Apples.ipynb)

### Tomatoe Disease

- [Tomato leaf disease](https://github.com/gubuktani/MachineLearning-GubukTani/blob/main/Notebook/Tomato_leaf_disease.ipynb)

### Potato Disease

- [Potato leaf disease](https://github.com/gubuktani/MachineLearning-GubukTani/blob/main/Notebook/Gubuktani_Potato_disease.ipynb)

### Rice Disease

- [Rice leaf disease](https://github.com/gubuktani/ml-models/blob/main/Notebook/Rice_disease.ipynb)

### Leaf Detection

- [Leaf detection](https://github.com/gubuktani/MachineLearning-GubukTani/blob/main/Notebook/leafDetection.ipynb)

## Prerequisites
1. [Jupyter Notebook](https://test-jupyter.readthedocs.io/en/latest/install.html) or [Google Colab](https://colab.research.google.com/)
2. Python (version 3.10 or higher) 
3. Tensorflow (version 2.12 or higher)
4. Pillow

## How to use
1. Download the model `.h5` file & `json` file
2. Import the necessary libraries in your Jupyter Notebook or Google Colab: 
  ```python
  import tensorflow as tf
  from PIL import Image
  import numpy as np
  import json
  ```
3. Load the model from the `.h5` file & `.json` file for class name:
  ```python
  model = tf.keras.models.load_model('path/to/model.h5')
  with open('/path/to/class.json', 'r') as file:
    class_name = json.load(file)
  ```
4. Prepare your image data for inference. Assuming you have an image file named `image.jpg` that you want to use, use the following code to load and preprocess the image:
  ```python
  image_path = 'path/to/image.jpg'
  image = np.array(Image.open(image_path).convert("RGB").resize((256, 256)))
  image = image / 255
  img_array = tf.expand_dims(image, 0)
  ```
5. Perform inference using the loaded model:
  ```python
  class_mapping = list(class_name)
  predictions = model.predict(image_array)
  predicted_class_index = np.argmax(predictions)
  predicted_class = class_mapping[predicted_class_index]

  print(predicted_class)
  ```
  
  The `predicted_class` variable will contain the predicted output based on the input image.

6. Customize the code as per your specific requirements.
