from dataset import *
from model import inference
from torch.utils.data import DataLoader
import os, argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help = "path to the data")
    parser.add_argument("--model_name", type = str, help = "model name to be tested")
    parser.add_argument("--cfg_path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    args = parser.parse_args()
    test_loader = get_dataset()

    os.makedirs("result", exist_ok = True)
    os.makedirs("record", exist_ok = True)

    match args.model_name:
        case "gemini_2p0_flash":
            print("=" * 5 + "evaluate gemini 2.0 flash" + "=" * 5)
            result, records = inference(args, test_loader)
            with open("result/gemini_2p0_flash.json", "w") as f:
                json.dump(result, f, indent = 2)
            with open("record/gemini_2p0_flash_record.json", "w") as f:
                json.dump(records, f, indent = 2)
    

        case "video_llama_13b":
            print("=" * 5 + "video_llama 13b" + "=" * 5)
            result, records = inference(args, test_loader)
            with open("result/video_llama_13b_1.json", "w") as f:
                json.dump(result, f, indent = 2)
            with open("record/video_llama_13b_record_1.json", "w") as f:
                json.dump(records, f, indent = 2)

        case "video_llama_7b":
            print("=" * 5 + "video_llama 7b" + "=" * 5)
            result, records = inference(args, test_loader)
            with open("result/video_llama_7b_1.json", "w") as f:
                json.dump(result, f, indent = 2)
            with open("record/video_llama_7b_record_1.json", "w") as f:
                json.dump(records, f, indent = 2)

if __name__ == "__main__":
    main()