gemini_2p0_flash:
	python main.py --data_path data --model_name gemini_2p0_flash

video_llama_13b:
	export PYTHONPATH=/data/plnguyen2908/AV_eval_pipeline/model/open/video_llama_1:$PYTHONPATH
	CUDA_VISIBLE_DEVICES=2 python main.py \
		--data_path data \
		--model_name video_llama_13b \
		--cfg_path /data/plnguyen2908/AV_eval_pipeline/model/open/video_llama_1/eval_configs/video_llama_eval_withaudio_13b.yaml \
		--model_type llama_v2

video_llama_7b:
	export PYTHONPATH=/data/plnguyen2908/AV_eval_pipeline/model/open/video_llama_1:$PYTHONPATH
	CUDA_VISIBLE_DEVICES=2 python main.py \
		--data_path data \
		--model_name video_llama_7b \
		--cfg_path /data/plnguyen2908/AV_eval_pipeline/model/open/video_llama_1/eval_configs/video_llama_eval_withaudio_7b.yaml \
		--model_type llama