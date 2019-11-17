from PIL import Image
from PIL import ImageFilter
import os
import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-id", "--input_dataset", required=True, help="Path to where the images resides")
ap.add_argument("-o", "--output_dataset", required=True, help="Path to where the video will be stored")
ap.add_argument("-f", "--factor", required=True, help="Augmentation factor")
args = vars(ap.parse_args())

path = args["input_dataset"]
output_folder = args["output_dataset"]
increase_factor = int(args["factor"])

images_names = os.listdir(path)

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


for image_name in images_names:
    for i in range(increase_factor):
        img = Image.open(path + "/" + image_name)

        img = apply_transformation(img)
        image_name_split = image_name.split(".")
        img.save(output_folder + "/" + image_name_split[0] + "_" + str(i) + "." + image_name_split[1])
