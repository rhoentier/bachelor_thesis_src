import cv2
import numpy as np
import torch


def classify(model, image):
    network = model.model.net
    network.eval()

    if image.shape[1] != model.image_size:
        image = cv2.resize(image, (model.image_size, model.image_size),
                           interpolation=cv2.INTER_LANCZOS4)

    image = np.array(image).astype('float32')
    if np.max(image) > 1:
        image = image / 255.
    image = torch.Tensor(image)
    image = image.transpose(2, 1).transpose(1, 0).unsqueeze(0)

    output_original = network(image.to(model.model.device))
    return np.argmax(output_original.detach().cpu().numpy())
