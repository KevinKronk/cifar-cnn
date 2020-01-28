import tensorflow.keras as tfk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data(label_mode="coarse")
model = tfk.models.load_model('cifar.h5')


class_names = {0: 'aquatic mammals', 1: 'fish', 2: 'flowers', 3: 'food containers',
               4: 'fruit and vegetables', 5: 'household electrical devices',
               6: 'household furniture', 7: 'insects', 8: 'large carnivores',
               9: 'large man-made outdoor things', 10: 'large natural outdoor scenes',
               11: 'large omnivores and herbivores', 12: 'medium-sized mammals',
               13: 'non-insect invertebrates', 14: 'people', 15: 'reptiles', 16: 'small mammals',
               17: 'trees', 18: 'vehicles 1', 19: 'vehicles 2'}


plt.figure(figsize=(10, 6))
# Choose random images
test = np.random.randint(0, 9999, size=(8,))

sub = 1
for i in test:
    image = np.expand_dims(x_test[i], axis=0)
    image = tf.dtypes.cast(image, tf.float16)
    prediction_vals = model.predict(image)
    pred = np.argpartition(prediction_vals, -2)[:, -2:]
    y_true = np.int(y_test[i])
    plt.subplot(2, 4, sub)
    plt.imshow(x_test[i], cmap='binary')
    one = class_names[np.int(pred[:, 1])]
    two = class_names[np.int(pred[:, 0])]
    plt.xlabel(f"Top 1: {one}\nTop 2: {two}\
    \nTrue: {class_names[y_true]}")
    plt.xticks([])
    plt.yticks([])
    sub += 1
plt.show()
