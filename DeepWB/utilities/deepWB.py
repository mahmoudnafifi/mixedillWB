"""
 Deep white-balance editing main function (inference phase)
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import numpy as np
import torch
from torchvision import transforms
import DeepWB.utilities.utils as utls
from src import imresize


def deep_wb(image, task='all', net_awb=None, net_t=None, net_s=None,
            device='cpu', s=656):
    # check image size
    h, w, _ = image.shape
    image_resized = imresize.imresize(image, output_shape=((
        round(h / max(h, w) * s), round(w / max(h, w) * s))))
    h, w, _ = image_resized.shape
    if w % 2 ** 4 == 0:
        new_size_w = w
    else:
        new_size_w = w + 2 ** 4 - w % 2 ** 4

    if h % 2 ** 4 == 0:
        new_size_h = h
    else:
        new_size_h = h + 2 ** 4 - h % 2 ** 4

    inSz = (new_size_h, new_size_w)
    if not ((h, w) == inSz):
        image_resized = imresize.imresize(image_resized, output_shape=inSz)

    image = np.array(image)
    image_resized = np.array(image_resized)
    img = image_resized.transpose((2, 0, 1))
    #img = img / 255
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    if task == 'all':
        net_awb.eval()
        net_t.eval()
        net_s.eval()
        with torch.no_grad():
            output_awb = net_awb(img)
            output_t = net_t(img)
            output_s = net_s(img)

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        output_awb = tf(torch.squeeze(output_awb.cpu()))
        output_awb = output_awb.squeeze().cpu().numpy()
        output_awb = output_awb.transpose((1, 2, 0))
        m_awb = utls.get_mapping_func(image_resized, output_awb)
        output_awb = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_awb))

        output_t = tf(torch.squeeze(output_t.cpu()))
        output_t = output_t.squeeze().cpu().numpy()
        output_t = output_t.transpose((1, 2, 0))
        m_t = utls.get_mapping_func(image_resized, output_t)
        output_t = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_t))

        output_s = tf(torch.squeeze(output_s.cpu()))
        output_s = output_s.squeeze().cpu().numpy()
        output_s = output_s.transpose((1, 2, 0))
        m_s = utls.get_mapping_func(image_resized, output_s)
        output_s = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_s))

        return output_awb, output_t, output_s

    elif task == 'awb':
        net_awb.eval()
        with torch.no_grad():
            output_awb = net_awb(img)

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        output_awb = tf(torch.squeeze(output_awb.cpu()))
        output_awb = output_awb.squeeze().cpu().numpy()
        output_awb = output_awb.transpose((1, 2, 0))
        m_awb = utls.get_mapping_func(image_resized, output_awb)
        output_awb = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_awb))

        return output_awb

    elif task == 'editing':
        net_t.eval()
        net_s.eval()
        with torch.no_grad():
            output_t = net_t(img)
            output_s = net_s(img)

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        output_t = tf(torch.squeeze(output_t.cpu()))
        output_t = output_t.squeeze().cpu().numpy()
        output_t = output_t.transpose((1, 2, 0))
        m_t = utls.get_mapping_func(image_resized, output_t)
        output_t = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_t))

        output_s = tf(torch.squeeze(output_s.cpu()))
        output_s = output_s.squeeze().cpu().numpy()
        output_s = output_s.transpose((1, 2, 0))
        m_s = utls.get_mapping_func(image_resized, output_s)
        output_s = utls.outOfGamutClipping(utls.apply_mapping_func(image, m_s))

        return output_t, output_s
