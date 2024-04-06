import pytorch_lightning as pl

from datamodule import TrafficLightModule
from model import EFFB3Model
from lib.config import CONF


# 初始化数据模块
data_module = TrafficLightModule(txt_file=CONF.dataset_tlc_classifier.labels_txt_path, batch_size=CONF.datamodule_tlc_classifier.batch_size)

# 初始化模型
model = EFFB3Model(freeze_layers=CONF.model_tlc_classifier.freeze_layers)

# 初始化 ModelCheckpoint 回调
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath='./saved_models/',
    filename='best_model',
    save_top_k=1,
    mode='min'
)

# 初始化 EarlyStopping 回调
early_stopping_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',  # 监视验证集损失
    patience=5,         # 忍耐次数，即多少个epoch没有改善时停止训练
    mode='min'           # 通过最小化验证集损失来判断是否停止
)

# 初始化 Trainer
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=300,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback, early_stopping_callback]  # 添加 ModelCheckpoint early_stopping_callback 回调
)

# 训练模型
trainer.fit(model, data_module)

# tensorboard --logdir=/home/jiachen/TOD2D/TLC/lightning_logs

