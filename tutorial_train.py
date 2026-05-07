from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

# Configs
config_path = './models/cldm_v15.yaml'
resume_path = '/NAS_data/yjy/Controlnet_2026_0504/models/control_sd15_ini.ckpt'
batch_size = 2
logger_freq = 800
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config_path).cpu()
state_dict = load_state_dict(resume_path, location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f'Missing keys when loading checkpoint: {missing_keys}')
print(f'Unexpected keys when loading checkpoint: {unexpected_keys}')
color_branch_prefixes = (
    'control_model.color_input_blocks.',
    'control_model.color_zero_convs.',
    'control_model.color_middle_block.',
    'control_model.color_middle_block_out.',
)
if not any(key.startswith(color_branch_prefixes) for key in state_dict):
    initialized_color_branch = model.control_model.initialize_color_branch_from_control_branch()
    if initialized_color_branch:
        print(
            'Initialized newly added color ControlNet branch from the loaded hint branch. '
            'It will be included in future ModelCheckpoint files.'
        )
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
config = OmegaConf.load(config_path)
dataset_params = OmegaConf.to_container(config.get('data', {}).get('params', {}), resolve=True)
dataset = MyDataset(**dataset_params)
dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(
                filename='epoch_{epoch:03d}-step_{step}',
                dirpath='/NAS_data/hjf/ControlNet-main/checkpoints_color_hints/sar2opt',
                every_n_epochs=10,
                save_weights_only=False,
                save_top_k=-1  # 保存所有检查点，不限制数量
            )
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback])


# Train!
trainer.fit(model, dataloader)
