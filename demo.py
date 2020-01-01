from argparse import ArgumentParser
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import random
import cv2
import os

parser = ArgumentParser()
parser.add_argument('--image_dir', default='/Users/hasty/Downloads/lfw', type=str, help='Directory of images')


def main():
    args = parser.parse_args()

    model = keras.models.load_model('models/generator.h5')
    print(model.summary())
    inputs = keras.Input((None, None, 4))
    outputs = model(inputs)
    model = keras.models.Model(inputs, outputs)

    # Surrogate file reader:
    image_paths = []
    for root, dir, file in os.walk(args.image_dir):
        if file[0].endswith('.jpg'):
            image_paths.append(os.path.join(root, file[0]))

    # image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]
    random.shuffle(image_paths)
    # Select images to plot, max 6
    n_images = min(len(image_paths), 4)
    fig, ax = plt.subplots(n_images, 6, figsize=(20, 10))
    for idx in range(n_images):
        for col in range(6):
            image = cv2.imread(image_paths[idx], 1)
            image = cv2.resize(image, (512, 512))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image / 255) * 2 - 1
            h, w, _ = image.shape
            if col == 0:
                ax[idx, col].imshow((image + 1.0) / 2.0)
                ax[idx, col].set_title('Original')
            else:
                condition = np.ones(shape=[1, h, w, 1]) * (col - 1)
                conditioned_images = np.concatenate([np.expand_dims(image, axis=0), condition], axis=-1)
                aged_image = model.predict(conditioned_images)[0]
                ax[idx, col].imshow((aged_image + 1.0) / 2.0)
    plt.show()


if __name__ == '__main__':
    main()
