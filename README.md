

Local implementation of local JEV like models 

models:

https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer/tree/main
https://huggingface.co/convaiinnovations/laya

install uv:
macos/linux -> curl -LsSf https://astral.sh/uv/install.sh | sh
windows -> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

install ext modules:
uv sync

run:
uv run python src/jev/laya/run.py

train:
uv run python src/jev/laya/train.py --dataset_path src/jev/laya/example_dataset.json --output_dir ./my_laya_model --epochs 4

bench using typed-decisions HF dataset:
uv run python src/jev/laya/test.py --model_path ./my_laya_model --compare_base

bench using custom dataset:
uv run python src/jev/laya/test.py --model_path ./my_laya_model --dataset_path src/jev/laya/benchmark.json

enjoy 
:)


