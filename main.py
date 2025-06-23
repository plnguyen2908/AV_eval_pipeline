from dataset import *
from model import *
from torch.utils.data import DataLoader
import os, argparse, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help = "path to the data")
    parser.add_argument("--model_name", type = str, help = "model name to be tested")
    args = parser.parse_args()
    test_loader = get_dataset()

    os.makedirs("result", exist_ok = True)

    if args.model_name == "gemini_2p0_flash":
        print("=" * 5 + "evaluate gemini 2.0 flash" + "=" * 5)
        result = inference(args, test_loader)
        with open("result/gemini_2p0_flash.json", "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()