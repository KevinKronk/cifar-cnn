import tensorflow.keras as tfk
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data(label_mode="coarse")
# x_train = np.reshape(x_train, [60000, 28, 28, 1])
# x_test = np.reshape(x_test, [10000, 28, 28, 1])
model = tfk.models.load_model('cifar.h5')

# class_names = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
#                5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

class_names = {0: 'aquatic mammals', 1: 'fish', 2: 'flowers', 3: 'food containers',
               4: 'fruit and vegetables', 5: 'household electrical devices',
               6: 'household furniture', 7: 'insects', 8: 'large carnivores',
               9: 'large man-made outdoor things', 10: 'large natural outdoor scenes',
               11: 'large omnivores and herbivores', 12: 'medium-sized mammals',
               13: 'non-insect invertebrates', 14: 'people', 15: 'reptiles', 16: 'small mammals',
               17: 'trees', 18: 'vehicles 1', 19: 'vehicles 2'}

plt.figure(figsize=(10, 6))
test = np.random.randint(0, 9999, size=(8,))
print(test)
sub = 1
for i in test:
    # image = x_test[i]
    image = np.expand_dims(x_test[i], axis=0)
    image = tf.dtypes.cast(image, tf.float16)
    # image = np.reshape(image, [1, 28, 28, 1])
    prediction_vals = model.predict(image)
    pred = np.argpartition(prediction_vals, -5)[:, -5:]
    #prediction = np.argmax(prediction_vals)
    y_true = np.int(y_test[i])
    plt.subplot(2, 4, sub)
    plt.imshow(x_test[i], cmap='binary')
    one = class_names[np.int(pred[:,4])]
    two = class_names[np.int(pred[:,3])]
    three = class_names[np.int(pred[:,2])]
    four = class_names[np.int(pred[:,1])]
    five = class_names[np.int(pred[:,0])]
    # , {two}, {three}, {four}, {five}
    plt.xlabel(f"Top 1: {one}\nTop 2: {two}\
    \nTrue: {class_names[y_true]}")
    plt.xticks([])
    plt.yticks([])
    sub += 1
plt.show()
