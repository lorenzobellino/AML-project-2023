import wandb
from config.baseline import *
from config.transform import *
from config.federated import *


def setup(step, args):
    wandb.login()
    if step == 2:
        transformer_dictionary = {
            "random-horizontal-flip": RANDOM_HORIZONTAL_FLIP,
            "color-jitter": COLOR_JITTER,
            "random-rotation": RANDOM_ROTATION,
            "random-crop": RANDOM_CROP,
            "random-vertical-flip": RANDOM_VERTICAL_FLIP,
            "central-crop": CENTRAL_CROP,
            "random-resized-crop": RANDOM_RESIZE_CROP,
            "resize": RESIZE,
        }

        config = {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "step_size": STEP_SIZE,
            "gamma": GAMMA,
            "transformers": transformer_dictionary,
        }
        name = f"Step_2_{PARTITION}_lr{LR}_bs{BATCH_SIZE}_e{NUM_EPOCHS}"
    elif step == 3:
        config = {
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "momentum": MOMENTUM,
            "num_epochs": NUM_EPOCHS,
            "n_client": CLIENT_PER_ROUND,
            "round": N_ROUND,
            "tot_client": TOT_CLIENTS,
        }
        name = (
            f"Step_3_{PARTITION}_split{SPLIT}_rounds{N_ROUND}_clients{CLIENT_PER_ROUND}"
        )
    elif step == 4:
        if args.pretrain:
            transformer_dictionary = {
                "random-horizontal-flip": RANDOM_HORIZONTAL_FLIP,
                "color-jitter": COLOR_JITTER,
                "random-rotation": RANDOM_ROTATION,
                "random-crop": RANDOM_CROP,
                "random-vertical-flip": RANDOM_VERTICAL_FLIP,
                "central-crop": CENTRAL_CROP,
                "random-resized-crop": RANDOM_RESIZE_CROP,
                "resize": RESIZE,
            }

            config = {
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "num_epochs": NUM_EPOCHS,
                "step_size": STEP_SIZE,
                "gamma": GAMMA,
                "transformers": transformer_dictionary,
            }
            name = f"Step_4_pretrain_lr{LR}_bs{BATCH_SIZE}_e{NUM_EPOCHS}"
        else:
            config = {
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "momentum": MOMENTUM,
                "num_epochs": NUM_EPOCHS,
                "n_client": CLIENT_PER_ROUND,
                "round": N_ROUND,
                "tot_client": TOT_CLIENTS,
            }
            name = f"Step_4_FDA_P{PARTITION}"
            # raise NotImplementedError
    else:
        raise NotImplementedError
    wandb.init(
        project=f"STEP{step}",
        # entity="lor-bellino",
        config=config,
        name=name,
    )
