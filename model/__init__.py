from .closed import *
from .open_model import *
import os, tqdm, textwrap, ast, re
# moviepy < 2.0
# from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
# moviepy > 2.0
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import shutil
import time
import random
random.seed(0)
import json

result = {}
records = {}

answer_prefixes = [
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is"
    "The correct option is",
    "Best answer:"
    "Best option:",
    "Answer:",
    "Option:",
    "The correct answer",
    "The correct option",
    "Based",
    "Correct answer",
    "\u261e",
    "<|im_end|>"
]

def extract_characters_regex(s):
    if s is None:
        c = ['A', 'B', 'C', 'D'][random.randint(0,3)]
        return c
    
    s = s.strip()
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    pattern = r"[.,:!'\";/\?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]"
    
    s = re.sub(pattern, " ", s)

    parsed = s.split()
    # print(parsed)
    matches = None
    for char in parsed:
        if char == "A" or char == "B" or char == "C" or char == "D" or char == "E":
            matches = char
            break

    if matches is None:
        return ""
    return matches[0]

def process_answer(gt, pred, question):
    if gt == pred:
        result[f'level 3: {question["task_id"]}']["matched"] += 1
        result[f'level 2: {question["sub_category"]}']["matched"] += 1
        result[f'level 1: {question["category"]}']["matched"] += 1
    
    result[f'level 3: {question["task_id"]}']["accuracy"] = round(result[f'level 3: {question["task_id"]}']["matched"] / result[f'level 3: {question["task_id"]}']["total"] * 100, 2)
    result[f'level 2: {question["sub_category"]}']["accuracy"] = round(result[f'level 2: {question["sub_category"]}']["matched"] / result[f'level 2: {question["sub_category"]}']["total"] * 100, 2)
    result[f'level 1: {question["category"]}']["accuracy"] = round(result[f'level 1: {question["category"]}']["matched"] / result[f'level 1: {question["category"]}']["total"] * 100, 2)

    return (gt == pred)

def inference(args, dataset):
    temporary_dir = os.path.join(os.getcwd(), args.temp_dir)
    try:
        shutil.rmtree(temporary_dir)
    except Exception as e:
        pass

    os.makedirs(temporary_dir, exist_ok = True)

    if args.model_name == "video_llama_13b" or args.model_name == "video_llama_7b":
        chat = model_init(args)
    elif args.model_name == "video_llama2_7b":
        model, processor, tokenizer = video_llama2_model_init()
    elif "panda" in args.model_name:
        model = pandagpt_model_init(args)
    elif "phi" in args.model_name:
        model, processor, generation_config = phi4_model_init(args)
    elif "onellm" in args.model_name:
        model = onellm_model_init(args)
    elif "uio" in args.model_name:
        model, processor = uio2_model_init(args)
    elif "NExTGPT" in args.model_name:
        model = NExTGPT_model_init(args)
    elif "AnyGPT" in args.model_name:
        model = anygpt_model_init(args)
    elif args.model_name == "vita1":
        model, audio_processor, image_processor, tokenizer = vita1_model_init(args)
    elif args.model_name == "vita1_5":
        model, audio_processor, image_processor, tokenizer = vita1_5_model_init(args)
    elif "xblip" in args.model_name:
        model = xblip_model_init(args)
    elif "Qwen2.5" in args.model_name:
        model, processor = qwen2_5Omni_model_init(args)
    elif "stream-omni" in args.model_name:
        model, tokenizer, image_processor, cosyvoice = streamomni_model_init(args)
    elif "ola" in args.model_name:
        model, tokenizer, image_processor = ola_model_init(args)
    elif "Qwen3" in args.model_name:
        model, processor = qwen3omni_model_init(args)

    cnt = 0
    wrong_cnt = 0

    recorded = {}

    if "gemini-2.5-pro" in args.model_name or "Qwen3-Omni" in args.model_name:
        if os.path.exists(f"./record/{args.model_name}_record_{args.task_id}_audio_{args.audio}_visual_{args.visual}.json"):
            with open(f"./record/{args.model_name}_record_{args.task_id}_audio_{args.audio}_visual_{args.visual}.json", "r") as f:
                recorded = json.load(f)

    for idx, question in tqdm.tqdm(enumerate(dataset), total = len(dataset)):
        original_video_path = os.path.join(args.data_path, question["video_path"])
        original_audio_path = os.path.join(args.data_path, question["audio_path"])

        while True:
            try:

                # uncomment if you use moviepy > 2.0
                with VideoFileClip(original_video_path) as video:
                    max_duration = video.duration
                    mm, ss = map(int, question["start_time"].split(":"))
                    start_secs = mm * 60 + ss
                    mm, ss = map(int, question["end_time"].split(":"))
                    end_secs = min(mm * 60 + ss, max_duration)

                    if start_secs > end_secs:
                        print(question)
                    
                    new_video_path = os.path.join(temporary_dir, f"video_{start_secs}-{end_secs}.mp4")
                    new_audio_path = os.path.join(temporary_dir, f"audio_{start_secs}-{end_secs}.wav")
                    new_combined_path = os.path.join(temporary_dir, f"video_audio_{start_secs}-{end_secs}.mp4")

                    video_sub = video.subclipped(start_secs, end_secs)
                    video_sub.write_videofile(
                        new_video_path, 
                        codec="libx264", 
                        audio=False,
                        logger = None
                    )

                    with AudioFileClip(original_audio_path) as audio:

                        audio_sub = audio.subclipped(start_secs, end_secs)
                        audio_sub.write_audiofile(
                            new_audio_path,
                            logger = None
                        )

                        composite_audio = CompositeAudioClip([audio_sub])
                        video_with_audio = video_sub.with_audio(composite_audio)
                        video_with_audio.write_videofile(
                            new_combined_path,
                            codec="libx264",       # codec video
                            audio_codec="aac",     # codec audio
                            fps=video.fps
                        )
                break

            except Exception as e:
                print(f"found error: {e} for {question}. Retry cutting video")
                time.sleep(5)
                continue

        choices = ast.literal_eval(question["choices"])
        choices_str = "\n".join(choices)

        question_prompt = f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n{question['question']}\n{choices_str}\nThe best answer is:"
    
        # break
        result[f'level 1: {question["category"]}'] = result.get(f'level 1: {question["category"]}', {})
        result[f'level 1: {question["category"]}']["matched"]= result[f'level 1: {question["category"]}'].get("matched", 0)
        result[f'level 1: {question["category"]}']["total"] = result[f'level 1: {question["category"]}'].get("total", 0) + 1

        result[f'level 2: {question["sub_category"]}'] = result.get(f'level 2: {question["sub_category"]}', {})
        result[f'level 2: {question["sub_category"]}']["matched"]= result[f'level 2: {question["sub_category"]}'].get("matched", 0)
        result[f'level 2: {question["sub_category"]}']["total"]= result[f'level 2: {question["sub_category"]}'].get("total", 0) + 1

        result[f'level 3: {question["task_id"]}'] = result.get(f'level 3: {question["task_id"]}', {})
        result[f'level 3: {question["task_id"]}']["matched"]= result[f'level 3: {question["task_id"]}'].get("matched", 0)
        result[f'level 3: {question["task_id"]}']["total"]= result[f'level 3: {question["task_id"]}'].get("total", 0) + 1
        
        # add more if needed for ablation
        id = question["question_id"]

        ans = ""
        if "gemini" in args.model_name:
            if id in recorded and recorded[id]["question_prompt"] == question_prompt:
                print(f"{id} exists")
                ans = recorded[id]["llm response"]
            elif args.audio:
                # print(new_audio_path)
                ans = gemini_process(new_audio_path, question_prompt, args.model_name, idx)
            elif args.visual:
                # print(new_video_path)
                ans = gemini_process(new_video_path, question_prompt, args.model_name, idx)
            else:
                ans = gemini_process(new_combined_path, question_prompt, args.model_name, idx)
        elif args.model_name == "video_llama_13b" or args.model_name == "video_llama_7b":
            ans = video_llama_process(chat, new_combined_path, question_prompt)
        elif args.model_name == "video_llama2_7b":
            ans = video_llama2_process(new_combined_path, question_prompt, model, processor, tokenizer)
            # print(ans)
        elif args.model_name == "pandagpt_7b" or args.model_name == "pandagpt_13b":
            ans = pandagpt_process(model, question_prompt, new_audio_path, new_video_path, 512, [])
        elif args.model_name == "phi4":
            ans = phi4_process(new_audio_path, new_video_path, args.num_frames, question_prompt, model = model, processor = processor, generation_config = generation_config)
        elif args.model_name == "onellm":
            ans = onellm_process(model, new_audio_path, new_video_path, question_prompt)
        elif args.model_name ==  "uio2-large" or args.model_name == "uio2-xl" or args.model_name == "uio2-xxl":
            ans = uio2_process(model, processor, new_combined_path, question_prompt)
        elif args.model_name ==  "NExTGPT":
            ans = NExTGPT_process(model, new_video_path, new_audio_path, question_prompt)
            ans = ans[0]
        elif args.model_name == "AnyGPT":
            ans = anygpt_process(model, new_video_path, new_audio_path, question_prompt)
        elif args.model_name == "vita1":
            ans = vita1_process(model, audio_processor, image_processor, tokenizer, new_video_path, new_audio_path, question_prompt)
        elif args.model_name == "vita1_5":
            ans = vita1_5_process(model, audio_processor, image_processor, tokenizer, new_video_path, new_audio_path, question_prompt)
        elif "xblip" in args.model_name:
            ans = xblip_process(model, new_audio_path, new_video_path, question_prompt)
        elif "Qwen2.5" in args.model_name:
            ans = qwen2_5Omni_process(model, processor, new_combined_path, question_prompt)
        elif "stream-omni" in args.model_name:
            ans = streamomni_process(model, tokenizer, image_processor, cosyvoice, new_audio_path, new_video_path, question_prompt)
        elif "ola" in args.model_name:
            ans = ola_process(model, tokenizer, image_processor, new_video_path, new_audio_path, question_prompt)
        elif "Qwen3" in args.model_name:
            if id in recorded and recorded[id]["question_prompt"] == question_prompt:
                print(f"{id} exists")
                ans = recorded[id]["llm response"]
            else:
                media_type = "video" if args.visual else ("audio" if args.audio else "both")
                media_path = new_video_path if args.visual else (new_audio_path if args.audio else new_combined_path)
                ans = qwen3omni_process(model, processor, media_type, media_path, question_prompt)
            
        
        records[id] = {
            "video_id": question["video_id"],
            "task_id": question["task_id"],
            "question_prompt": question_prompt,
            "answer": question["answer"],
            "llm response": ans
        }

        print(ans)
        print(question["answer"])
        
        ans = extract_characters_regex(ans)
        print(ans)

        if ans != question["answer"]:
            wrong_cnt += 1
        cnt += 1

        print(f"current error rate: {wrong_cnt / cnt}")

        records[id]["parsed llm answer"] = ans

        # process_answer(choices[ord(question["answer"]) - ord("A")], ans, question)
        matched = process_answer(question["answer"], ans, question)

        records[id]["matched"] = matched

        os.remove(new_audio_path)
        os.remove(new_video_path)
        os.remove(new_combined_path)
    
    print(result)
    return result, records