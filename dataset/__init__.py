from huggingface_hub import snapshot_download
from .get_dataset import get_dataset

snapshot_download("lvwerra/stack-exchange-paired", repo_type="dataset",revision='main')