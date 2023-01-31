import cv2
import numpy as np
import torch
from torch.autograd import Variable


###
#
# Taken from:
# Johannes Alecke. Analyse und Optimierung von Angriffen auf tiefe neuronale Netze, Hochschule Bonn-Rhein-Sieg, 2020
#
###

def calculate_saliency_map(model_list, original_image, target_label, percentile: float):
    saliency_maps = []

    for model in model_list:
        network = model.model.net
        network.eval()
        image = original_image

        if np.max(image) > 1:
            image = image.astype(np.float32) / 255
        if image.shape[1] != model.image_size:
            image = cv2.resize(image, (model.image_size, model.image_size),
                               interpolation=cv2.INTER_LANCZOS4)
        image = torch.Tensor(image)
        image = image.transpose(2, 1).transpose(1, 0).unsqueeze(0)

        image = image.to(model.model.device)
        grads = smooth_grad(network, image, target_label,
                            sample_size=16, percent_noise=10)
        saliency_map = get_saliency_map_from_grads(grads)
        saliency_maps.append(saliency_map)

    mean_saliency_map = None
    for saliency_map in saliency_maps:
        if mean_saliency_map is None:
            mean_saliency_map = saliency_map
        else:
            mean_saliency_map += saliency_map

    mean_saliency_map = mean_saliency_map / np.max(mean_saliency_map)
    mean_saliency_map_sorted = np.sort(np.reshape(mean_saliency_map, (-1)))
    index_p = int(percentile * mean_saliency_map_sorted.shape[0])
    threshold = mean_saliency_map_sorted[index_p]
    saliency_map_threshold = mean_saliency_map > threshold
    colored_saliency_map = np.zeros(saliency_map_threshold.shape)
    colored_saliency_map = np.repeat(np.expand_dims(
        colored_saliency_map, axis=-1), 3, axis=-1)
    colored_saliency_map[:, :, 0] = saliency_map_threshold
    colored_saliency_map[:, :, 1] = saliency_map_threshold
    colored_saliency_map[:, :, 2] = saliency_map_threshold
    colored_saliency_map = cv2.resize(colored_saliency_map, dsize=(original_image.shape[0], original_image.shape[1]),
                                      interpolation=cv2.INTER_NEAREST)
    colored_saliency_map_negated = np.where(colored_saliency_map != 1)
    original_image_small_values = np.where(original_image < (255 / 2))
    original_image_big_values = np.where(original_image > (255 / 2))
    image_with_saliency_map = np.copy(original_image)
    image_with_saliency_map[original_image_small_values] = 255
    image_with_saliency_map[original_image_big_values] = 0
    image_with_saliency_map[colored_saliency_map_negated] = original_image[colored_saliency_map_negated]

    return image_with_saliency_map


# Code verändert nach: https://github.com/sar-gupta/convisualize_nb
#
# Aus Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg.
# Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825, 2017.
def smooth_grad(net, tensor_input, label, sample_size=10, percent_noise=10):
    final_grad = torch.zeros(
        (1, 3, tensor_input.shape[-1], tensor_input.shape[-1])).cuda()

    for i in range(sample_size):
        temp_input = tensor_input

        noise = torch.from_numpy(
            np.random.normal(loc=0, scale=(
                    percent_noise * (tensor_input.cpu().detach().max() - tensor_input.cpu().detach().min()) / 100),
                             size=temp_input.shape)).type(torch.cuda.FloatTensor)

        temp_input = (temp_input + noise)
        temp_input = Variable(temp_input, requires_grad=True)

        output = net.forward(temp_input)
        output[0][label].backward()
        final_grad += temp_input.grad.data

    grads = final_grad / sample_size
    grads = grads.clamp(min=0)
    grads.squeeze_()
    grads.transpose_(0, 1)
    grads.transpose_(1, 2)
    grads = np.amax(grads.cpu().numpy(), axis=2)

    return grads


def get_saliency_map_from_grads(grads):
    image = grads
    if type(image) == torch.Tensor:
        image = image.permute(1, 2, 0).cpu().detach()
    return image
