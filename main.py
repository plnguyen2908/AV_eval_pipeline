from dataset import *
from model import inference
from torch.utils.data import DataLoader
import os, argparse, json
import sys
# proj_root = os.path.abspath(os.path.join(__file__, "..", "model", "open", "LAVIS_XInstructBLIP"))
# if proj_root not in sys.path:
#     sys.path.insert(0, proj_root)
# print("\n".join(sys.path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str, required = True, help = "model name to be tested")
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
    parser.add_argument("--num_frames", type=int, help="the number of frames for phi4")
    parser.add_argument("--task_id", type=str, default = None)
    parser.add_argument("--category", type=str, default = None)
    parser.add_argument("--sub_category", type=str, default = None)
    parser.add_argument("--audio", dest="audio", action='store_true')
    parser.add_argument("--visual", dest="visual", action='store_true')
    parser.add_argument("--temp_dir", type=str, default = "temp")
    parser.add_argument("--data_path", type=str, default = "/data/plnguyen2908/AV_eval_pipeline/data", help = "original path to downloaded data")
    args = parser.parse_args()
    test_loader = get_dataset(category = args.category, sub_category=args.sub_category, task_id = args.task_id, )

    os.makedirs("result", exist_ok = True)
    os.makedirs("record", exist_ok = True)


    result, records = inference(args, test_loader)

    if "gemini" in args.model_name:
        print("=" * 5 + f"evaluate {args.model_name}" + "=" * 5)
        with open(f"result/{args.model_name}.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/{args.model_name}_record_{args.task_id}_audio_{args.audio}_visual_{args.visual}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "video_llama_13b":
        print("=" * 5 + "video_llama 13b" + "=" * 5)
        with open("result/video_llama_13b.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/video_llama_13b_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "video_llama_7b":
        print("=" * 5 + "video_llama 7b" + "=" * 5)
        with open("result/video_llama_7b.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/video_llama_7b_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)
    
    elif args.model_name == "video_llama2_7b":
        print("=" * 5 + "video_llama2 7b" + "=" * 5)
        with open("result/video_llama2_7b.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/video_llama2_7b_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "pandagpt_7b":
        print("=" * 5 + "pandagpt 7b" + "=" * 5)
        with open("result/pandagpt_7b.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/pandagpt_7b_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "pandagpt_13b":
        print("=" * 5 + "pandagpt 13b" + "=" * 5)
        with open("result/pandagpt_13b.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/pandagpt_13b_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "phi4":
        print("=" * 5 + "phi4" + "=" * 5)
        with open("result/phi4.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/phi4_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)
    
    elif args.model_name == "onellm":
        print("=" * 5 + "onellm" + "=" * 5)
        with open("result/onellm.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/onellm_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "uio2-large":
        print("=" * 5 + "uio2-large" + "=" * 5)
        with open("result/uio2-large.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/uio2-large_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "uio2-xl":
        print("=" * 5 + "uio2-xl" + "=" * 5)
        with open("result/uio2-xl.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/uio2-xl_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "uio2-xxl":
        print("=" * 5 + "uio2-xxl" + "=" * 5)
        with open("result/uio2-xxl.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/uio2-xxl_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)
        
    elif args.model_name == "NExTGPT":
        print("=" * 5 + "NExTGPT" + "=" * 5)
        with open("result/NExTGPT.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/NExTGPT_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)
    
    elif args.model_name == "AnyGPT":
        print("=" * 5 + "AnyGPT" + "=" * 5)
        with open("result/AnyGPT.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/AnyGPT_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "vita1":
        print("=" * 5 + "VITA 1.0" + "=" * 5)
        with open("result/vita1.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/vita1_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "vita1_5":
        print("=" * 5 + "VITA 1.5" + "=" * 5)
        with open("result/vita1_5.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/vita1_5_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "xblip_7b":
        print("=" * 5 + "XBlipInstruct 7b" + "=" * 5)
        with open("result/xblip_7b.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/xblip_7b_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "xblip_13b":
        print("=" * 5 + "XBlipInstruct 13b" + "=" * 5)
        with open("result/xblip_13b.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/xblip_13b_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "Qwen2.5-Omni-3B":
        print("=" * 5 + "Qwen2.5-Omni-3B" + "=" * 5)
        with open("result/Qwen2.5-Omni-3B.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/Qwen2.5-Omni-3B_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "Qwen2.5-Omni-7B":
        print("=" * 5 + "Qwen2.5-Omni-7B" + "=" * 5)
        with open("result/Qwen2.5-Omni-7B.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/Qwen2.5-Omni-7B_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "stream-omni":
        print("=" * 5 + "StreamOmni" + "=" * 5)
        with open("result/StreamOmni.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/StreamOmni_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif args.model_name == "ola":
        print("=" * 5 + "Ola" + "=" * 5)
        with open("result/ola.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/ola_record_{args.task_id}.json", "w") as f:
            json.dump(records, f, indent = 2)

    elif "Qwen3" in args.model_name:
        print("=" * 5 + args.model_name + "=" * 5)
        with open(f"result/{args.model_name}.json", "w") as f:
            json.dump(result, f, indent = 2)
        with open(f"record/{args.model_name}_record_{args.task_id}_audio_{args.audio}_visual_{args.visual}.json", "w") as f:
            json.dump(records, f, indent = 2)


import os


if __name__ == "__main__":
    os.environ['HF_HOME'] = '/data/plnguyen2908/cache'
    main()