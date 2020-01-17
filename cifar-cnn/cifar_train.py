from tensorflow import keras as tfk

# Load cifar 100 data
(x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data(label_mode='coarse')

model = tfk.Sequential()
model.add(tfk.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation=tfk.activations.relu,
                        input_shape=(32, 32, 3), padding='same'))
model.add(tfk.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation=tfk.activations.relu,
                        padding='same'))
model.add(tfk.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tfk.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                            activation=tfk.activations.relu, padding='same'))
model.add(tfk.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        activation=tfk.activations.relu))
model.add(tfk.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tfk.layers.Flatten())
model.add(tfk.layers.Dense(1024, activation=tfk.activations.relu))
model.add(tfk.layers.Dropout(0.5))
model.add(tfk.layers.Dense(256, activation=tfk.activations.relu))
model.add(tfk.layers.Dropout(0.5))
model.add(tfk.layers.Dense(20, activation=tfk.activations.softmax))


model.compile(optimizer=tfk.optimizers.Adadelta(),
              loss=tfk.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)


print(loss, accuracy)

model.save('cifar.h5')
print('Model Saved.')
