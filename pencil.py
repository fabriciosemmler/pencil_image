import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

'''This program transforms an image into a pencil sketch simulation.'''


def pencil(id, display=True):
    # Get the original image identification
    name = id

    # Read image
    image = cv2.imread('originals/' + name)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert grayscale image to enhance details
    inverted_image = 255 - gray_image

    # Create the pencil sketch
    blurred = cv2.GaussianBlur(inverted_image, (111, 111), 0)
    inverted_blurred = 255 - blurred
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    # Save the sketch
    cv2.imwrite('sketches/sketch_' + name, pencil_sketch)

    # Display the results
    if display:
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(14, 8))
        plt.subplot(1, 2, 1)
        plt.title('Original image', size=18)
        plt.imshow(RGB_img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Sketch', size=18)
        rgb_sketch = cv2.cvtColor(pencil_sketch, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_sketch)
        plt.axis('off')
        plt.show()


# Capture in a list all files in the originals directory
onlyfiles = [f for f in listdir('originals/') if isfile(join('originals/', f))]

# Transform in pencil sketch all files in the originals directory
for file in onlyfiles:
    pencil(file)
