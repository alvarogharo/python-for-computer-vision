from PIL import Image
import os
import numpy

path = "./sequences"
image_prefix = "frame"
image_suffix = ".jpg"
output_folder = "./output/"
h_range = [29, 88]
s_range = [43, 255]
v_range = [126, 255]
blend_weight = 0.2
images = []

num_images = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

for i in range(num_images):
    images.append(Image.open(path + "/" + image_prefix + str(i+1) + image_suffix))

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def convert_and_threshold(image):
    result = image.convert("HSV")
    original_image = result.copy()
    result = numpy.array(result)
    h = result[:, :, 0]
    s = result[:, :, 1]
    v = result[:, :, 2]
    mask = (h >= h_range[0]) & (h < h_range[1]) & \
           (s >= s_range[0]) & (s < s_range[1]) & \
           (v >= v_range[0]) & (v < v_range[1])
    result = result * mask[:, :, None]
    result = Image.fromarray(result, "HSV")
    result = Image.blend(result, original_image, blend_weight)
    return result.convert("RGB")


for i in range(len(images)):
    print(str(round(i/len(images)*100, 2)) + "%")
    images[i] = convert_and_threshold(images[i])
    images[i].save(output_folder + image_prefix + str(i+1) + image_suffix)
