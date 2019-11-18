import cfg
from . import unet


def create(model_name):
    if model_name == 'unet':
        model = unet.UNet(n_class=cfg.n_class)
    else:
        raise RuntimeError(f'Unidentified model name <{model_name}>')

    return model.to(cfg.device)
