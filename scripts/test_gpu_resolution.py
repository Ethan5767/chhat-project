"""Quick test to check what accelerator PTL resolves on the current system."""
import torch
import os

print("CUDA_AVAIL=" + str(torch.cuda.is_available()))
print("DEVICE_COUNT=" + str(torch.cuda.device_count()))
print("VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"))
if torch.cuda.is_available():
    print("GPU_NAME=" + torch.cuda.get_device_name(0))

from rfdetr import RFDETRMedium

m = RFDETRMedium()
cfg = m.get_train_config(dataset_dir="datasets/cigarette_packs", epochs=1)
print("ACCELERATOR=" + str(cfg.accelerator))
print("AMP=" + str(m.model_config.amp))
print("COMPILE=" + str(m.model_config.compile))
print("DEVICE_CFG=" + str(m.model_config.device))

from rfdetr.training import build_trainer

t = build_trainer(cfg, m.model_config)
print("TRAINER_ACC=" + t.accelerator.__class__.__name__)
print("TRAINER_PRECISION=" + str(t.precision))
print("TRAINER_DEVICES=" + str(t.device_ids))
print("TRAINER_NUM_DEVICES=" + str(t.num_devices))
