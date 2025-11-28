from huggingface_hub import snapshot_download
# from .get_dataset import get_dataset

local_dir = "/data/plnguyen2908/AV-SpeakerBench/data"
snapshot_download("plnguyen2908/Holistic_AVQA_bench", repo_type="dataset", local_dir=local_dir, local_dir_use_symlinks=False)