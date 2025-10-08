label:
	docker run -it -p 8080:8080 -v $(pwd)/editor:/label-studio/data heartexlabs/label-studio:latest

sync:
	uv sync --preview-features extra-build-dependencies

train:
	uv run --preview-features extra-build-dependencies src/main.py --train_json assets/coco/multiple-set/result.json --val_json assets/coco/multiple-set/result.json --img_dir assets/coco/multiple-set/images --epochs 1 --batch_size 2