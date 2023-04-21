import time
import importlib
import logging
import argparse
from argparse import RawTextHelpFormatter

# import warnings

# from utils import parse_args, modify_command_options

logger = logging.getLogger("FED_SS")


def run_experiment(args, logger):
    if args.step == 1:
        logger.info("Generating the Dataset for cityscapes")
        main_module = "dataset_generation.main"
    elif args.step == 2:
        logger.info("Centralized baseline of Cityscapes")
        main_module = "centralized.main"
    elif args.step == 3:
        logger.info("Federated + Semantic Segmentation")
        main_module = "federated.main"

    elif args.step == 4:
        logger.info("Domain Adaptation")
        main_module = "centralized.main"

    elif args.step == 5:
        logger.info("Pseudo Labeling")
        main_module = "federated.main"
    else:
        raise NotImplementedError

    main = getattr(importlib.import_module(main_module), "main")
    main(args, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Federated Learning with semantic segmentation",
        description="Based on the choosen step the program will perform different actions.",
        epilog="Choose a Step and run the program.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        help="Step to run:\n"
        + "\t1: Generating the Dataset for cityscapes\n"
        + "\t2: Centralized baseline\n"
        + "\t3: Federated + Semantic Segmentation\n"
        + "\t4: Domain Adaptaion:\n",
        required=True,
    )
    parser.add_argument(
        "-dts",
        "--dataset",
        type=str,
        help="Dataset to use: GTA (GTA5) or CTSC (Cityscapes)",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--load",
        type=str,
        help="Load the model from previous run",
    )
    parser.add_argument(
        "-net",
        "--network",
        type=str,
        help="Network to use: <bisenet> OR <mobilenet>",
        required=True,
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    logger.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG if args.debug else logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    start = time.time()

    run_experiment(args, logger)

    end = time.time()
    logger.info(f"Elapsed time: {round(end - start, 2)}")
