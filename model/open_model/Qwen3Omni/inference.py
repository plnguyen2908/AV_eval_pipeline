from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import torch

use_audio_in_video = True
return_audio = False


def qwen3omni_model_init(args):
    model_type = args.model_name.split("-")[-1]
    MODEL_PATH = f"/data/plnguyen2908/AV_eval_pipeline/model/open_model/Qwen3Omni/Qwen3-Omni-30B-A3B-{model_type}"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        MODEL_PATH,
    )

    return model, processor

def qwen3omni_process(model, processor, media_type, media_path, question):

    conversation = [
        {
            "role": "user",
            "content": [],
        },
    ]

    print(media_type, media_path)

    if media_type == "video":
        conversation[0]["content"].append({"type": "video", "video": media_path})
        use_audio_in_video = False
    elif media_type == "audio":
        conversation[0]["content"].append({"type": "audio", "audio": media_path})
        use_audio_in_video = False
    else:
        conversation[0]["content"].append({"type": "video", "video": media_path})
        use_audio_in_video = True
                
    conversation[0]["content"].append({"type": "text", "text": question})

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=use_audio_in_video)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids, audio = model.generate(**inputs, 
                                    thinker_return_dict_in_generate=True,
                                    thinker_max_new_tokens=1024, 
                                    thinker_do_sample=False,
                                    speaker="Ethan", 
                                    use_audio_in_video=use_audio_in_video,
                                    return_audio=return_audio)
    
    text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)

    print(text)
    if "</think>" in text[0]:
        text[0] = text[0].split("</think>")[-1]

    print(text)
    return text[0]