import time
import importlib
import argparse

# import warnings

# from utils import parse_args, modify_command_options


def run_experiment():
    # if args.framework == "federated":
    #     main_module = "fed_setting.main"
    #     main = getattr(importlib.import_module(main_module), "main")
    #     main(args)
    # elif args.framework == "centralized":
    #     main_module = "centr_setting.main"
    #     main = getattr(importlib.import_module(main_module), "main")
    #     main(args)
    # else:
    #     raise NotImplementedError
    time.sleep(2)


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(
        prog="Federated Learning with semantic segmentation",
        description="Based on the choosen step the program will perform different actions.",
        epilog="Choose a Step and run the program.",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        help="Step to run:\n\t1: Generating the Dataset for cityscapes\n\t2: Centralized baseline\n\t3: Federated + Semantic Segmentation",
        required=True,
    )

    args = parser.parse_args()

    print(f"Step {args.step} selected -> {type(args.step)}")
    run_experiment()

    end = time.time()
    print(f"Elapsed time: {round(end - start, 2)}")
