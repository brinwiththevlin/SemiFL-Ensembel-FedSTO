import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, PoseModel, OBBModel, ClassificationModel
from ultralytics.models import yolo
from ultralytics import YOLO
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.FATAL)


class SRIPLoss(v8DetectionLoss):
    def __init__(self, model: DetectionModel, srip_lambda=0.001):
        super().__init__(model)
        self.srip_lampda = srip_lambda
        self.model = model
        
    def __call__(self, pred, target):
        original_loss, loss_items = super().__call__(pred, target)
        srip_loss = self.srip_loss()
        total_loss = original_loss + srip_loss*self.srip_lambda

        return total_loss, loss_items

    def srip_loss(self):
        srip_loss = 0
        W = 0
        for name, param in self.model.named_parameters():
            if "conv" in name:
                W += 1
                w = param.view(param.size(0), -1)
                wt = torch.transpose(w, 0, 1)
                if w.shape[0] > w.shape[1]:
                    srip_loss += self.sigma(w@wt - torch.eye(w.shape[1]))**2
                else:
                    srip_loss += self.sigma(wt@w - torch.eye(w.shape[0]))**2

        return srip_loss/W if W else 0

    def sigma(self, x: torch.Tensor):
        # v is a n dimentional vector randomly initialized with normal distribution
        v = torch.randn(x.shape[0], 1)
        return torch.norm(x**4@v)/torch.norm(x**3@v)
                

class orthoDetectionModel(DetectionModel):
    def __init__(self, model: YOLO, srip_lambda=0.001):
        super().__init__(model)
        self.srip_lambda = srip_lambda
        
    def init_criterion(self):
        return SRIPLoss(self, self.srip_lambda) 


class orthoYolo(YOLO):
    def __init__(self, srip_lambda=0.001, **kwargs):
        super().__init__(**kwargs)
        self.srip_lambda = srip_lambda

    @property
    def task_map(self):
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": orthoDetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class SSODYolo(YOLO):
    def __init__(self, model_state_dict=None, selective=False, orthogonal=False, **kwargs):
        assert not (selective and orthogonal), "selective and orthogonal cannot be True at the same time"
        super().__init__(**kwargs)
        if orthogonal:
            self.model = orthoYolo(pretrained=False, **kwargs)
            self.freeze = False
        elif selective:
            self.model = YOLO(pretrained=False, **kwargs)
            self.freeze = list(range(11, 25))
        else: 
            self.model = YOLO(pretrained=False, **kwargs)
            self.freeze = False
        
        if model_state_dict:    
            self.model.load_state_dict(model_state_dict)

    def train(self, data, epochs, imgsz, verbose=False):
        results = self.model.train(data, epochs, imgsz, verbose, freeze=self.freeze)
        state =  self.model.state_dict() 
        return state

        
if __name__ == "__main__":
    model = orthoYolo( model="yolov8n.pt")
    # results = model.train(data="coco8.yaml", epochs=100, imgsz=640, verbose=False)
    for name, param in model.named_parameters():
        print(name, param.shape)