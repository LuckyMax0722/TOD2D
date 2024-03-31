import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models

from torch.optim.lr_scheduler import StepLR


# 初始化 Lightning 模型
class EFFB3Model(pl.LightningModule):
    def __init__(self, freeze_layers=False):
        super().__init__()
        self.model = models.efficientnet_b3(weights='DEFAULT')

        # 冻结前面的所有层
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False

        # 解冻最后分类层的参数
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        #  change the last output FC layer
        self.model.classifier = nn.Sequential(
            self.model.classifier,
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)  # 每个epoch后，学习率乘0.9
        return [optimizer], [scheduler]


if __name__ == '__main__':
    m = EFFB3Model()
    for name, module in m.named_modules():
        print(name, module)
