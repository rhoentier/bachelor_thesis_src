import datetime
import os
import uuid
from itertools import combinations
from os.path import exists

import cv2
from ai_framework.saliency_map import calculate_saliency_map
from image_framework.image_framework import load_base_image, display_image, generate_image, SignImages, \
    load_sign_image, sign_image_path, remove_background, remove_background_with_original_alpha
from commercial_system.commercial_system_classifier import classify_image
from ai_framework.models.model_loader import load_self_trained_inception_net3, \
    load_self_trained_inception_net3_gray, load_pretrained_inception_v3, load_pretrained_mobile_net_v2, \
    load_pretrained_resnext
from webcam.webcam import Webcam
from webcam.webcam_classifier import classify

to_gtsrb = {
    2: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    9: 7,
    11: 8,
}


class HilFramework:

    def __init__(self, base_image):
        self.networks = [load_self_trained_inception_net3(), load_self_trained_inception_net3_gray(),
                         load_pretrained_resnext(), load_pretrained_inception_v3(), load_pretrained_mobile_net_v2()]

        base_name = f"{os.path.dirname(__file__)}/logs/"
        csv_base_name = f"{os.path.dirname(__file__)}/logs/csv/"
        csv_extension = ".csv"
        date_time = datetime.datetime.now().strftime("%d:%m:%Y, %H:%M:%S")
        saliency_map_base_name = f"{os.path.dirname(__file__)}/logs/saliency_maps/"
        log_uuid = uuid.uuid4()

        if not os.path.exists(base_name):
            os.mkdir(base_name)
        if not os.path.exists(csv_base_name):
            os.mkdir(csv_base_name)
        if not os.path.exists(saliency_map_base_name):
            os.mkdir(saliency_map_base_name)
        while os.path.exists(f"{csv_base_name}run_{log_uuid}{csv_extension}"):
            log_uuid = uuid.uuid4()

        self.csv_file_path = f"{csv_base_name}{date_time}_{log_uuid}{csv_extension}"
        self.saliency_map_base_path = saliency_map_base_name
        self.csv = open(self.csv_file_path, "a+")
        self.csv.write(
            "image_class,image_id,phase,system,"
            "parameter,parameter_value,result,combination\n")

        self.base_image_name = base_image
        self.base_image = load_base_image(base_image)
        display_image(self.base_image)

        sign = load_sign_image(
            sign_image_path[SignImages.KMH80.value] + "00.png")
        sign = remove_background(sign)
        display_image(
            generate_image(sign, self.base_image, self.base_image_name))

        self.webcam = Webcam()
        self.webcam.calibrate_cam()

    def __log_result(self, result, image_class=None, image_id=None, phase=None, system=None,
                     parameter=None, parameter_value=None, combination=None):
        combination_string = ""
        if combination is not None:
            for model_name in combination:
                combination_string += model_name.model.name + "|"
            combination_string = combination_string[:-1]
        self.csv.write(
            f"{image_class},{image_id},{phase},{system},{parameter},"
            f"{parameter_value},{result},{combination_string}\n")

    def classify_image_with_commercial_system(self, image):
        display_image(image)
        result = classify_image()[0]
        if result in to_gtsrb.keys():
            result = to_gtsrb[int(result)]
        else:
            result = 255
        display_image(self.base_image)

        return result

    def classify_image_with_cnn(self, image, model):
        display_image(image)
        webcam_image = self.webcam.get_image()
        result = classify(model, webcam_image)
        display_image(self.base_image)

        return result

    def classify_all_images(self):
        for sign_class_id in SignImages:
            signs_in_dict = len(os.listdir(
                sign_image_path[sign_class_id.value]))

            for counter in range(signs_in_dict):
                sign = load_sign_image(
                    sign_image_path[sign_class_id.value] + str(counter).zfill(2) + ".png")
                sign = remove_background(sign)
                image = generate_image(
                    sign, self.base_image, self.base_image_name)

                for model in self.networks:
                    result = self.classify_image_with_cnn(image, model)
                    self.__log_result(result=result, image_class=sign_class_id.value, image_id=str(counter).zfill(
                        2), phase="classification", system=model.model.name, parameter=None, parameter_value=None,
                                      combination=None)

                result = self.classify_image_with_commercial_system(image)
                self.__log_result(result=result, image_class=sign_class_id.value, image_id=str(counter).zfill(
                    2), phase="classification", system="commercial_system", parameter=None, parameter_value=None,
                                  combination=None)

    def saliency_map_attack(self):
        model_combinations = list()
        for i in range(1, len(self.networks) + 1):
            model_combinations += list(combinations(self.networks, i))
        for combination in model_combinations:
            flag = True
            for model in combination:
                for other_model in combination:
                    if model.image_size != other_model.image_size:
                        flag = False
                        break
                if flag is False:
                    break
            if flag is False:
                continue
            self.__run_saliency_map_loop(combination)

    def __run_saliency_map_loop(self, models):
        for sign_class_id in SignImages:
            signs_in_dict = len(os.listdir(
                sign_image_path[sign_class_id.value]))

            for counter in range(signs_in_dict):
                sign_id = str(counter).zfill(2)
                sign = load_sign_image(
                    sign_image_path[sign_class_id.value] + sign_id + ".png")
                original_sign = remove_background(sign)

                for i in range(5):
                    attacked_sign = calculate_saliency_map(
                        models, sign, sign_class_id.value, percentile=(1 - 0.01 - i / 100))
                    attacked_sign = remove_background_with_original_alpha(
                        attacked_sign, original_sign)
                    image = generate_image(
                        attacked_sign, self.base_image, self.base_image_name)
                    for model in self.networks:
                        result = self.classify_image_with_cnn(image, model)

                        if result != sign_class_id.value:
                            combination_string = ""
                            for model_name in models:
                                combination_string += model_name.model.name + "|"
                            combination_string = combination_string[:-1]
                            if not exists(
                                    self.saliency_map_base_path + f"{sign_class_id.value}_{sign_id}_"
                                                                  f"{combination_string}_{1 - 0.01 - i / 100}.png"):
                                cv2.imwrite(
                                    self.saliency_map_base_path + f"{sign_class_id.value}_{sign_id}_"
                                                                  f"{combination_string}_{1 - 0.01 - i / 100}.png",
                                    attacked_sign)

                        self.__log_result(result=result, image_class=sign_class_id.value, image_id=sign_id,
                                          phase="saliency_map",
                                          system=model.model.name, parameter="percentile",
                                          parameter_value=round(1 - 0.01 - i / 100, 2), combination=models)

                    result = self.classify_image_with_commercial_system(image)
                    self.__log_result(result=result, image_class=sign_class_id.value, image_id=sign_id,
                                      phase="saliency_map",
                                      system="commercial_system", parameter="percentile",
                                      parameter_value=round(1 - 0.01 - i / 100, 2), combination=models)
