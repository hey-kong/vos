import os
from PIL import Image

cropped_images_dir = "./cropped_images"


def crop_img(img_path, bbox, i):
    img = Image.open(img_path)
    cropped_img = img.crop(bbox)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(cropped_images_dir, f"cropped_image_{img_name}_{i}.jpg")
    cropped_img.save(save_path)
    return save_path
