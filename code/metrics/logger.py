# Simple tb logger
import torch

from exp import ex

'''
geometry_normalizer = {
    'cartesian': 4, # [0,1]x[0,1]x[0,1]x[0,1]
    'angular': 98.696,   # [-pi,pi]x[-.5pi,.5pi]x[0,2pi]x[0,pi]
    'spherical': 61.348, # [-1,1]x[-1,1]x[-1,1]x[0,2pi]x[0,pi]
    'quaternion': 17 # [0,1]x[-1,1]x[-1,1]x[0,2]x[0,2]
}
'''

def write_logs(logger, timestamp, lr, stat, meta, mode="train"):
    if mode == "train":
        logger.add_scalar('Train/lr', lr, timestamp)

        for k, v in stat.items():
            if type(v) == torch.Tensor and v.dim() == 0:
                logger.add_scalar(f'Train/{k}', v.item(), timestamp)
            elif type(v) == str:
                logger.add_text(f'Train/{k}', v, timestamp)

    else:
        for k, v in stat.items():
            if type(v) in [int, float]:
                logger.add_scalar(f'{mode.capitalize()}/{k}', v, timestamp)
            elif type(v) == torch.Tensor and v.dim() == 0:
                logger.add_scalar(f'{mode.capitalize()}/{k}', v.item(), timestamp)
            elif type(v) == str:
                logger.add_text(f'{mode.capitalize()}/{k}', v, timestamp)
        #logger.add_image('Eval/image', img, timestamp)


@ex.capture()
def adjust_grounding_error(error, geometry):
    return error * geometry_normalizer[geometry]