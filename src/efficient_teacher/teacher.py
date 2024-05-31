import time
from datetime import timedelta
from pathlib import Path

import torch.nn as nn
from EMA import ModelEMA, SemiSupModelEMA

from config import cfg


class SSODTrainer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.ema = ModelEMA(self.model)
        self.semi_ema = SemiSupModelEMA(self.ema.ema, cfg["ema_decay"])
        self.psuedo_label = FairPsuedoLabel()

        self.build_ddp_model()

    def build_ddp_model(self):
        raise NotImplementedError

    def train(self, callbacks):
        self.last_opt_step = -1
        self.results = (0, 0, 0, 0, 0, 0)
        self.best_fitness = 0

        for self.epoch in range(self.start_epoch, self.epochs):
            if self.epoch == self.break_epoch:
                break
            self.model.train()
            self.train_in_epoch(callbacks)
            self.after_epoch(callbacks, val)

        results = self.after_train(callbacks, val)
        return results


class FairPsuedoLabel:
    def __init__(self):
        self.nms_conf_threshold = cfg["efficient_teacher"]["nms_conf_threshold"]
        self.nms_iou_threshold = cfg["efficient_teacher"]["nms_iou_threshold"]
        self.multi_label = cfg["efficient_teacher"]["multi_label"]
        # self.names = cfg["Dataset"]["names"]
