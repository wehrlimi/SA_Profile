import os
import logging
import time

from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data import KneeDataset, LakeFSLoader, DataAugmentation
from models import UNet, CELoss, save_model
from scripts import run_evaluation, define_device

def train_model(cfg):
    writer = SummaryWriter(cfg.program.result_path) # for tensorboard

    OmegaConf.save(cfg, os.path.join(cfg.program.result_path, "parameter_config.yaml")) # save parameter settings

    device = define_device(cfg)

    data_loader_lakefs = LakeFSLoader(endpoint=cfg.lakefs.s3_endpoint,
                                      repo_name=cfg.lakefs.repository,
                                      local_cache_path=cfg.data.local_cache_path,
                                      ca_path=cfg.lakefs.ca_path)

    augmentation = None
    if cfg.training.use_augmentation:
        augmentation = DataAugmentation(cfg)

    data_set_training = KneeDataset(
        data_path_file=os.path.join(cfg.data.data_split_path, "train.json"),
        lakefs_loader=data_loader_lakefs,
        keypoint_names=cfg.data.landmark_names,
        image_size=cfg.program.image_size,
        augmentation=augmentation,
        keypoint_std=3.)

    data_set_validation = KneeDataset(
        data_path_file=os.path.join(cfg.data.data_split_path, "val.json"),
        lakefs_loader=data_loader_lakefs,
        keypoint_names=cfg.data.landmark_names,
        image_size=cfg.program.image_size,
        augmentation=None)
    
    num_landmarks = data_set_training.get_num_landmarks()
    
    # define models
    model = UNet(in_channels=1,
                 out_channels=num_landmarks,
                 down_channels=cfg.model.down_channels,
                 down_layers=cfg.model.down_layers,
                 up_channels=cfg.model.up_channels,
                 up_layers=cfg.model.up_layers,
                 bottleneck_channels=cfg.model.bottleneck_channels,
                 bottleneck_layers=cfg.model.bottleneck_layers,
                 dropout_p=cfg.model.dropout_p,
                 residual=cfg.model.residual,
                 keypoint_extraction=cfg.model.keypoint_extraction)
    
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : (1 - cfg.optimizer.lr_min) * cfg.optimizer.lr_decay**epoch + cfg.optimizer.lr_min)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epoch)

    loss_function = CELoss()

    logging.info(f"Number of samples in training data set: {len(data_set_training)}")
    logging.info(f"Number of samples in validation data set: {len(data_set_validation)}")
    logging.info(f"Landmark model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")


    data_loader = DataLoader(data_set_training, batch_size=cfg.program.data_loader.batch_size,
                             num_workers=cfg.program.data_loader.num_workers,
                             shuffle=True, pin_memory=True,
                             persistent_workers=cfg.program.data_loader.persistent_workers,
                             collate_fn=lambda x : x)

    if not os.path.exists(cfg.program.checkpoint_path):
        os.makedirs(cfg.program.checkpoint_path)

    # store the last value of the evaluation to save the best model
    last_eval_loss_value = float("inf")

    logging.info(f"Starting training")
    for epoch_idx in range(1, cfg.training.num_epoch+1):
        t1 = time.time()
        num_batches = len(data_loader)
        for run_idx, batch in enumerate(data_loader):
            iteration_idx = (epoch_idx - 1) * len(data_loader) + run_idx

            #
            # Train model
            #
            images = torch.stack([item['image'].to(device) for item in batch])
            heatmaps = torch.stack([item['heatmaps'].to(device) for item in batch])
            images.requires_grad_()

            pred_heatmaps = model(images)

            heatmaps_mask = ~ (torch.isnan(heatmaps).view(*heatmaps.shape[:-2], -1).any(dim=-1))
            
            loss = loss_function(pred_heatmaps[heatmaps_mask], heatmaps[heatmaps_mask])
            loss.backward()
            optimizer.step()

            # set all previous gradients of the model to zero; this is faster than model.zero_grad
            for param in model.parameters():
                param.grad = None
            
            #
            # Logging
            #
            writer.add_scalar(f'loss/cross-entropy', loss.item(), iteration_idx)

            logging.info(" ".join(["Epoch", str(epoch_idx)+f"/{cfg.training.num_epoch}",
                                   "iter", str(iteration_idx+1), str((iteration_idx+1) % num_batches)+f"/{num_batches}",
                                   "loss", "{:.2E}".format(loss.item(), 5)]))

        logging.info(f"lr = {scheduler.get_last_lr()}")
        scheduler.step()

        torch.cuda.empty_cache()

        t2 = time.time()
        logging.info(f'Epoch {epoch_idx} with {cfg.program.data_loader.num_workers} workers took {t2-t1} seconds.')

        if epoch_idx % cfg.training.eval_each_epoch == 0:
            eval_result = run_evaluation(cfg, data_set_validation, device, model, show_heatmaps=(epoch_idx in cfg.evaluation.create_images_epochs))
            logging.info(eval_result)
            writer.add_scalar('eval/keypoint', eval_result["pos_loss"], epoch_idx)
            writer.add_scalar('eval/keypoint large deviation', eval_result["pos_dev_loss"], epoch_idx)

        if epoch_idx % cfg.training.save_each_epoch == 0:
            # check if the model performance on the evaluation set improved
            if last_eval_loss_value > eval_result["pos_loss"]:
                last_eval_loss_value = eval_result["pos_loss"]
                save_model(cfg.program.checkpoint_path, model, "keypoint_model", epoch_idx)

    writer.close()
