from huggingface_hub import snapshot_download

REPO = "ghanaopenai/ghana-podcasts-twi-en"
REV = "70ac9aa9c94d09504fb01439a872ba461152b697"

path = snapshot_download(
    REPO,
    repo_type="dataset",
    revision=REV,
    allow_patterns=["data/*.parquet", "README.md"],
    local_dir="data/ghana-podcasts",
    max_workers=8,
)
print("downloaded to", path, flush=True)
