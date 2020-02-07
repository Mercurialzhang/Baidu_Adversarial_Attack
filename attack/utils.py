import numpy as np

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def tensor2img(tensor):
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))

    img = tensor.copy()

    img *= img_std
    img += img_mean
    return img  # range (0, 1)


def img2tensor(img):
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))

    tensor = img.copy()

    tensor -= img_mean
    tensor /= img_std
    return tensor


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def linf_img_tenosr(o, adv, epsilon=16.0 / 256):
    o_img = tensor2img(o)
    adv_img = tensor2img(adv)

    clip_max = np.clip(o_img + epsilon, 0., 1.)
    clip_min = np.clip(o_img - epsilon, 0., 1.)

    adv_img = np.clip(adv_img, clip_min, clip_max)

    adv_img = img2tensor(adv_img)

    return adv_img