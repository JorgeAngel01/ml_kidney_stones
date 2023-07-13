import helpers.transferLearningBaseModel as tlm
import torchvision.models as models
import torch.nn as nn

class InceptionModel(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001
        #LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(InceptionModel, self).__init__(hparams, seed=seed)
        self.inception = models.inception_v3(pretrained=pretrained, init_weights=True, aux_logits=False)
        #self.inception.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, num_classes))
        self.inception.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 256))
        self.inception.fc2 = nn.Sequential(nn.Linear(256, num_classes))
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        

    def forward(self, x):
        return self.inception(x)