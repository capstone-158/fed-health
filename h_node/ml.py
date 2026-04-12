import os
from shared.models import AdaptedAlexNet, AdaptedResNet18, ImprovedNet
from shared.state_manager import get_hospital_current_round
from shared.config import GLOBAL_WEIGHTS_PATH, NUM_CLASSES

def get_model():
    model_type = os.getenv("model_type")
    if model_type == "resnet":
        return AdaptedResNet18(NUM_CLASSES)
    elif model_type == "alexnet":
        return AdaptedAlexNet(NUM_CLASSES)
    elif model_type == "net":
        return ImprovedNet(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_global_weights():
    round_num = get_hospital_current_round()

    if round_num == 0:
        model = get_model()
        return model.state_dict()
    else:
        if not os.path.exists(GLOBAL_WEIGHTS_PATH):
            raise FileNotFoundError(f"{GLOBAL_WEIGHTS_PATH} not found")
        return torch.load(GLOBAL_WEIGHTS_PATH, map_location="cpu")
