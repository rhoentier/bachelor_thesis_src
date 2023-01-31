from hil_framework.hil_framework import HilFramework
from image_framework.image_framework import BaseImages


###
#
# To start run this file
#
###


def run_hil():
    hil_framework = HilFramework(base_image=BaseImages.LAND_2)

    hil_framework.classify_all_images()
    hil_framework.saliency_map_attack()


if __name__ == "__main__":
    run_hil()
