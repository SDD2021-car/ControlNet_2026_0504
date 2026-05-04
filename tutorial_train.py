from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint


# Configs
resume_path = '/data/hjf/TextDiff/ControlNet-main/models/control_sd15_ini.ckpt'
batch_size = 2
logger_freq = 800
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(
                filename='epoch_{epoch:03d}-step_{step}',
                dirpath='/NAS_data/hjf/ControlNet-main/checkpoints/GF3',
                every_n_epochs=10,
                save_weights_only=False,
                save_top_k=-1  # 保存所有检查点，不限制数量
            )
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback])


# Train!
trainer.fit(model, dataloader)
