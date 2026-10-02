import os
import sys
import logging

from omegaconf import OmegaConf
import torch
import numpy as np

from scripts import train_model, run_test


def main():
    # configuration
    cli_conf = OmegaConf.from_cli()
    if cli_conf.config is not None:
        cfg_conf = OmegaConf.load(cli_conf.config)
        cfg = OmegaConf.merge(cfg_conf, cli_conf)
    else:
        cfg = cli_conf

    # set the seeds if needed for torch and numpy
    if cfg.seeds.torch != -1:
        torch.manual_seed(cfg.seeds.torch)

    if cfg.seeds.numpy != -1:
        np.random.seed(cfg.seeds.numpy)
    
    if "gpu_id" in cli_conf:
        cfg.device.gpu_id = cli_conf.gpu_id


    # define the logger
    os.makedirs(cfg.program.result_path, exist_ok=True)
    #log_filename = os.path.join(cfg.program.result_path, "program.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
                        #handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler(sys.stdout)])


    if cfg.program.mode == "train":
        logging.info("Performing model training")
        train_model(cfg)
    elif cfg.program.mode == "infer":
        logging.error("Inference mode is not available through main.py.")
        logging.info("Please use scripts/two_stage_inference.py for two-stage inference.")
        logging.info("Example: python scripts/two_stage_inference.py --config config.yaml --sagittal_model <path> --axial_model <path> --input_volume <path>")
        raise NotImplementedError("Use scripts/two_stage_inference.py for inference")
    elif cfg.program.mode == "test":
        logging.info("Performing model test")
        run_test(cfg)
    else:
        raise KeyError(" ".join(("Mode", cfg.mode, "is not valid operation mode")))


if __name__ == "__main__":
    main()
