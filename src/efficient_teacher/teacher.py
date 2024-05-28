from config import cfg


class SSODTrainer:
    def __init__(self, callbacks, LOCAL_RANK, RANK, WORLD_SIZE):
        self.set_env(LOCAL_RANK, RANK, WORLD_SIZE, callbacks)
        self.psuedo_label = FairPsuedoLabel()


class FairPsuedoLabel:
    def __init__(self):
        self.nms_conf_threshold = cfg["SSOD"]["nms_conf_threshold"]
        self.nms_iou_threshold = cfg["SSOD"]["nms_iou_threshold"]
        self.multi_label = cfg["SSOD"]["multi_label"]
        self.names = cfg["Dataset"]["names"]
        self.num_points = cfg["Dataset"]["np"]
