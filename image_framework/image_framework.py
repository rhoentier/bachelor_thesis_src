import os
from enum import Enum

import cv2
import numpy as np
import screeninfo


class BaseImages(Enum):
    LAND_1 = 0
    LAND_2 = 1
    STADT_1 = 2


base_image_path = {
    0: f"{os.path.dirname(__file__)}/base_images/landstraße1.jpg",
    1: f"{os.path.dirname(__file__)}/base_images/landstraße2.jpg",
    2: f"{os.path.dirname(__file__)}/base_images/stadtstraße1.jpg",
}


class SignImages(Enum):
    KMH30 = 1
    KMH50 = 2
    KMH60 = 3
    KMH70 = 4
    KMH80 = 5
    KMH100 = 7
    KMH120 = 8


sign_image_path = {
    1: f"{os.path.dirname(__file__)}/sign_images/1/",
    2: f"{os.path.dirname(__file__)}/sign_images/2/",
    3: f"{os.path.dirname(__file__)}/sign_images/3/",
    4: f"{os.path.dirname(__file__)}/sign_images/4/",
    5: f"{os.path.dirname(__file__)}/sign_images/5/",
    7: f"{os.path.dirname(__file__)}/sign_images/7/",
    8: f"{os.path.dirname(__file__)}/sign_images/8/",
}


def calc_size(base_img):
    scale_factor = 1
    width = base_img.shape[1]
    height = base_img.shape[0]
    monitor = screeninfo.get_monitors()[1] if len(
        screeninfo.get_monitors()) == 2 else screeninfo.get_monitors()[0]
    if width > height:
        factor = (monitor.width * scale_factor) / width
    else:
        factor = (monitor.height * scale_factor) / height
    new_width = int(width * factor)
    new_height = int(height * factor)
    new_size = (new_width, new_height)
    return new_size


def load_base_image(image_id):
    return cv2.imread(base_image_path[image_id.value])


def load_sign_image(path_to_sign):
    return cv2.imread(path_to_sign)


def display_image(image):
    win_name = "image_window"
    # set image to second screen
    img_pos_y = 0
    sec_monitor = screeninfo.get_monitors()[1] if len(screeninfo.get_monitors()) == 2 else \
        screeninfo.get_monitors()[0]
    if sec_monitor is not None:
        img_pos_x = sec_monitor.x
    else:
        img_pos_x = 0
    # show window and move to second screen
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, img_pos_x, img_pos_y)
    cv2.setWindowProperty(
        win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    image = resize_image(image)
    cv2.imshow(win_name, image)
    cv2.waitKey(200)


def generate_image(sign, base_img, base_img_name):
    sign = cv2.resize(sign, (350, 350))

    if base_img_name == BaseImages.LAND_2:
        combined_image = add_image_at_position(base_img, sign, 3340, 1900)
    elif base_img_name == BaseImages.LAND_1:
        combined_image = add_image_at_position(base_img, sign, 2600, 800)
    elif base_img_name == BaseImages.STADT_1:
        combined_image = add_image_at_position(base_img, sign, 3940, 1500)
    else:
        combined_image = add_image_at_position(base_img, sign, 3340, 1900)

    combined_image = resize_image(combined_image)

    return combined_image


def resize_image(base_img):
    monitor = screeninfo.get_monitors()[1] if len(
        screeninfo.get_monitors()) == 2 else screeninfo.get_monitors()[0]
    new_size = calc_size(base_img)
    base_img = cv2.resize(base_img, new_size)
    height, width, channels = base_img.shape
    if height > monitor.height:
        base_img = base_img[height - monitor.height:height][0:width]
    elif width > monitor.width:
        base_img = base_img[0:height][width - monitor.width:width]
    return base_img


def add_image_at_position(background, foreground, x_offset=None, y_offset=None):
    new_background = np.copy(background)
    bg_h, bg_w, bg_channels = new_background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    has_alpha = fg_channels == 4
    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and new_background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    new_background_subsection = new_background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    if has_alpha:
        alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

        # construct an alpha_mask that matches the image shape
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
    else:
        alpha_mask = 1

    # combine the new_background with the overlay image weighted by alpha
    composite = new_background_subsection * \
                (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the new_background image that has been updated
    new_background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return new_background


def remove_background(image):
    t_lower = 50
    t_upper = 100
    edge_image = cv2.Canny(image, t_lower, t_upper)
    circles = cv2.HoughCircles(edge_image, cv2.HOUGH_GRADIENT, 1.6, 40, minRadius=int((image.shape[0] * 0.5) * 0.5),
                               maxRadius=int((image.shape[0] * 1.2) * 0.5))
    if circles is not None:
        circle = circles[0][0]
        if circle is not None:
            channels = cv2.split(image)
            if len(channels) == 3:
                b_channel, g_channel, r_channel = channels
                alpha_channel = np.ones(
                    b_channel.shape, dtype=b_channel.dtype) * 50
            else:
                b_channel, g_channel, r_channel, alpha_channel = channels
            image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            circle = np.round(circle[:]).astype("int")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask = cv2.circle(mask, (circle[0], circle[1]), int(
                circle[2] * 1.02), (255, 255, 255), -1, 8, 0)
            image[:, :, 3] = mask

    return image


def remove_background_with_original_alpha(image, original):
    if original.shape[2] == 4:
        original_alpha = cv2.split(original)[3]
        channels = cv2.split(image)
        if len(channels) == 3:
            b_channel, g_channel, r_channel = channels
        else:
            b_channel, g_channel, r_channel, alpha_channel = channels
        image = cv2.merge((b_channel, g_channel, r_channel, original_alpha))
    return image
