import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models

from torch.optim.lr_scheduler import StepLR


# 初始化 Lightning 模型
class EFFB3Model(pl.LightningModule):
    def __init__(self, freeze_layers=False, num_class_color=4, num_class_direction=5):
        super().__init__()
        self.model = models.efficientnet_b3(weights='DEFAULT')

        # 冻结前面的所有层
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False

        # 将最后的分类器层设为空Sequential，即弃置掉
        self.model.classifier = nn.Sequential()

        # 添加自定义的颜色分类器和方向分类器
        self.color_classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_class_color)
        )

        self.direction_classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_class_direction)
        )

    def forward(self, x):
        features = self.model(x)
        color_logits = self.color_classifier(features)
        direction_logits = self.direction_classifier(features)
        return color_logits, direction_logits

    def training_step(self, batch, batch_idx, ):
        x, color_y, direction_y = batch
        color_logits, direction_logits = self(x)

        color_train_loss = torch.nn.functional.cross_entropy(color_logits, color_y)
        self.log('color_train_loss', color_train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        direction_train_loss = torch.nn.functional.cross_entropy(direction_logits, direction_y)
        self.log('direction_train_loss', direction_train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        total_loss = color_train_loss + direction_train_loss
        self.log('train_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, color_y, direction_y = batch
        color_logits, direction_logits = self(x)

        color_loss = torch.nn.functional.cross_entropy(color_logits, color_y)
        direction_loss = torch.nn.functional.cross_entropy(direction_logits, direction_y)

        self.log('val_color_loss', color_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_direction_loss', direction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        total_loss = color_loss + direction_loss
        self.log('val_loss', total_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)  # 每个epoch后，学习率乘0.9
        return [optimizer], [scheduler]


if __name__ == '__main__':
    m = EFFB3Model()
    for name, module in m.named_modules():
        print(name, module)
