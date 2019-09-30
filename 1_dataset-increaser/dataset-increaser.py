from PIL import Image
from PIL import ImageFilter
import os.path
import random

path = "./test"
image_prefix = "test_"
image_suffix = ".JPEG"
output_folder = "./increased_test/"
images = []
increase_factor = 5

num_images = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

for i in range(num_images):
    images.append(Image.open(path + "/" + image_prefix + str(i) + image_suffix))

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def apply_transformation(image):
    transformation = random.randint(0, 3)
    result = 0
    if transformation == 0:
        rand_blur = random.randint(2, 10)
        result = image.filter(ImageFilter.GaussianBlur(rand_blur))
    elif transformation == 1:
        rand_resize = random.uniform(0.25, 2.5)
        result = image.resize((int(image.width * rand_resize), int(image.height * rand_resize)))
    else:
        result = image.transpose(Image.FLIP_TOP_BOTTOM)
    return result


for i in range(len(images)):
    for j in range(increase_factor):
        images[i] = apply_transformation(images[i])
        images[i].save(output_folder + image_prefix + str(i) + "," + str(j) + image_suffix)
