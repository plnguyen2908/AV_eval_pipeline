from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, default_conversation, conv_llava_llama_2
# Bắt buộc import module chứa VideoLLAMA để decorator chạy
import torch, decord
from omegaconf import OmegaConf
decord.bridge.set_bridge('torch')

def model_init(args):
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model_config, model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    return chat


def video_llama_process(chat, video_path, question):
    chat_state = conv_llava_llama_2.copy()
    img_list = []
    chat.upload_video(video_path, chat_state, img_list)
    chat.ask(question, chat_state)

    num_beams = 1
    temperature = 1.0

    # trả về tuple (reply, ...) nên lấy phần tử [0]
    reply = chat.answer(conv=chat_state,
                        img_list=img_list,
                        num_beams=num_beams,
                        temperature=temperature,
                        max_new_tokens=300,
                        max_length=2000)[0]
    return reply

    