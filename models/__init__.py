from models import encoder
from models import losses
from models import resnet

from models import noisy_models

REGISTERED_MODELS = {
    'finetune': noisy_models.FineTune,
    'mwnet': noisy_models.MWNetModel,
}
