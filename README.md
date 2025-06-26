# 🧠 Transfer Learning: Classificação de Gatos vs Cachorros

Projeto prático de aprendizado de máquina com **Transfer Learning**, desenvolvido durante a disciplina de Deep Learning na **Infinity School**, sob orientação do **Professor Gabriel**.  
Aluna: **Nayara Ventura**

---

## 🎯 Objetivo

Aplicar **Transfer Learning com MobileNet** para resolver o problema clássico de classificação binária entre **gatos e cachorros** a partir de imagens, utilizando **redes neurais convolucionais (CNNs)**.

---

## 📁 Base de Dados

- Fonte: [Kaggle - Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)
- 8.000 imagens para treinamento
- 2.000 imagens para teste

---

## 🔧 Tecnologias Utilizadas

- Python
- TensorFlow e Keras
- MobileNet (pré-treinada no ImageNet)
- Google Colab

---

## 🚀 Etapas do Projeto

### 📦 1. Importação das Bibliotecas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
```

### 🧠 2. Criação da Rede com Transfer Learning

Usamos o modelo **MobileNet** sem a última camada (`include_top=False`) e adicionamos novas camadas:

```python
model = MobileNet(weights='imagenet', include_top=False)

x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(50, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(inputs=model.input, outputs=preds)
```

### 🔒 3. Congelamento das Camadas

Somente as camadas adicionadas serão treinadas:

```python
for layer in model.layers[:88]:
    layer.trainable = False
for layer in model.layers[88:]:
    layer.trainable = True
```

---

## 🖼️ 4. Preparação das Imagens

Usando `ImageDataGenerator` com aumento de dados:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.4,
    zoom_range=0.4,
    height_shift_range=0.3,
    width_shift_range=0.3,
    rotation_range=50,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'catsxdogs/training_set',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'catsxdogs/test_set',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

---

## 🎓 5. Treinamento do Modelo

Compilando e treinando por 10 épocas:

```python
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    training_set,
    steps_per_epoch=250,
    epochs=10,
    validation_data=test_set,
    validation_steps=62
)
```

---

## 💾 6. Salvando e Baixando o Modelo

```python
model.save('catsxdogs_mobilenet.h5')

from google.colab import files
files.download('catsxdogs_mobilenet.h5')
```

---

## 🔍 7. Previsão com Imagens Novas

Realize a previsão de uma imagem nova:

```python
test_image = image.load_img('catsxdogs/single_prediction/cat_or_dog_1.jpg', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255

result = model.predict(test_image)
prediction = 'dog' if result[0][0] > 0.5 else 'cat'
print(prediction)
```

---

## 📌 Conclusão

Esse projeto demonstra a eficácia do **Transfer Learning** para tarefas de classificação de imagens, mesmo com um número limitado de épocas de treino e um conjunto de dados relativamente pequeno.

---

**Aluna:** Nayara Ventura  
**Professor:** Gabriel  
**Instituição:** Infinity School
