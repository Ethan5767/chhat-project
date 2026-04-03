"""Verify CHHAT_DATA_ROOT + finetuned DINO checkpoint (run on server)."""
import os
import sys

os.environ["CHHAT_DATA_ROOT"] = "/opt/chhat-data"
sys.path.insert(0, "/opt/chhat-project/backend")
import pipeline  # noqa: E402

print("module", pipeline.__file__)
print("_DATA_ROOT", pipeline._DATA_ROOT)
p = pipeline.DINO_FINETUNED_FULL_PATH
print("DINO_FINETUNED_FULL_PATH", p)
print("exists", p.is_file())
import torch  # noqa: E402

st = torch.load(p, map_location="cpu", weights_only=True)
n = len([k for k in st if str(k).startswith("dino.")])
print("dino_prefixed_keys", n)
