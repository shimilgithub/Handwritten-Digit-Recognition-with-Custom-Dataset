"""used for testing the images"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mnist import Mnist

def load_image(path):
    """Preprocess the image to black background, white digit"""
    #NOTE : since the image is already in 28 * 28 size and grayscale mode, there is no need of further processing
    img = Image.open(path)

    # image to numpy array
    img = np.array(img, dtype=np.float32)

    # Invert colors(black-> white and white -> black)
    img = 255 - img

    # Normalize to 0:255 range
    img = img - np.min(img)
    img = img * (255 / (np.max(img) - np.min(img)))

    # Normalize to 0:1  range
    img = img / 255.0

    #flatten image
    img = img.flatten()

    # Display image
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.show()

    return img

def main(image_path, expected_label):
    """ Main  function"""
    #create and instance for Mnist class and load network
    mnist_model = Mnist()
    mnist_model.init_network()

    # Load preprocessed image
    img = load_image(image_path)

    # Predict the digit
    y = mnist_model.predict(img)
    predicted_label = np.argmax(y)

    # Check if the prediction is correct
    if predicted_label == expected_label:
        print("\n\nSuccess: The image is correctly classified.\n")
    else:
        print(f"\n\nFail: Image {image_path} is for digit {expected_label} but the inference result is {predicted_label}\n")

if __name__ == "__main__":
    '''Call Main'''
    print("Instruction to run : Pass Two input arguments image path and digit of the image.\
        eg: $ python module5.py 3_2.png 3")
    image_path = sys.argv[1]
    expected_label = int(sys.argv[2])
    main(image_path, expected_label)
