from .closed import *
from .open import *
import os, tqdm,textwrap, ast
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
import shutil

result = {}
records = []

def process_answer(gt, pred, question):
    if gt in pred:
        result[question["task_id"]]["matched"] += 1
        result[question["sub_category"]]["matched"] += 1
        result[question["category"]]["matched"] += 1
    
    result[question["task_id"]]["accuracy"] = round(result[question["task_id"]]["matched"] / result[question["task_id"]]["total"] * 100, 2)
    result[question["sub_category"]]["accuracy"] = round(result[question["sub_category"]]["matched"] / result[question["sub_category"]]["total"] * 100, 2)
    result[question["category"]]["accuracy"] = round(result[question["category"]]["matched"] / result[question["category"]]["total"] * 100, 2)


def inference(args, dataset):
    temporary_dir = os.path.join(os.getcwd(), "temp")
    shutil.rmtree(temporary_dir)
    os.makedirs(temporary_dir, exist_ok = True)

    if args.model_name == "video_llama_13b" or args.model_name == "video_llama_7b":
        chat = model_init(args)

    for idx, question in tqdm.tqdm(enumerate(dataset), total = len(dataset)):
        original_video_path = os.path.join(args.data_path, question["video_path"])
        # original_audio_path = os.path.join(args.data_path, question["audio_path"])

        video = VideoFileClip(original_video_path)
        
        max_duration = video.duration
        mm, ss = map(int, question["start_time"].split(":"))
        start_secs = mm * 60 + ss
        mm, ss = map(int, question["end_time"].split(":"))
        end_secs = min(mm * 60 + ss, max_duration)
        
        new_video_path = os.path.join(temporary_dir, f"video_{start_secs}-{end_secs}.mp4")
        # new_audio_path = os.path.join(temporary_dir, f"audio_{start_secs}-{end_secs}.wav")
        # new_combined_path = os.path.join(
        #     temporary_dir,
        #     f"video_{start_secs}-{end_secs}_with_audio.mp4"
        # )

        video = video.subclipped(start_secs, end_secs)
        video.write_videofile(
            new_video_path, 
            codec="libx264", 
            audio_codec="aac",
            logger = None
        )
        video.close()

        # audio = AudioFileClip(original_audio_path).subclipped(start_secs, end_secs)
        # audio.write_audiofile(
        #     new_audio_path,
        #     logger = None
        # )
        # audio.close()

        # new_audioclip = CompositeAudioClip([audio])
        # video.audio = new_audioclip
        # video.write_videofile(
        #     new_combined_path,
        #     codec="libx264",
        #     audio_codec="aac",
        #     logger=None
        # )

        choices = ast.literal_eval(question["choices"])
        choices_str = "\n".join(choices)

        question_prompt = f"You are give a video and an audio, please answer the following question:\n{question['question']}\n{choices_str}\nAnswer with the letter and corresponding option from the given choices directly."

        result[question["category"]] = result.get(question["category"], {})
        result[question["category"]]["matched"]= result[question["category"]].get("matched", 0)
        result[question["category"]]["total"] = result[question["category"]].get("total", 0) + 1

        result[question["sub_category"]] = result.get(question["sub_category"], {})
        result[question["sub_category"]]["matched"]= result[question["sub_category"]].get("matched", 0)
        result[question["sub_category"]]["total"]= result[question["sub_category"]].get("total", 0) + 1

        result[question["task_id"]] = result.get(question["task_id"], {})
        result[question["task_id"]]["matched"]= result[question["task_id"]].get("matched", 0)
        result[question["task_id"]]["total"]= result[question["task_id"]].get("total", 0) + 1
        
        # add more if needed for ablation

        ans = ""
        match args.model_name:
            case "gemini_2p0_flash":
                ans = gemini_2p0_flash_process(new_video_path, question_prompt)
            case "video_llama_13b":
                ans = video_llama_process(chat, new_video_path, question_prompt)
            case "video_llama_7b":
                ans = video_llama_process(chat, new_video_path, question_prompt)
                # print(ans)

        process_answer(choices[ord(question["answer"]) - ord("A")], ans, question)
        records.append(
            {
                "task_id": question["task_id"],
                "question_prompt": question_prompt,
                "answer (letter)": question["answer"],
                "answer (gt to be compared)": choices[ord(question["answer"]) - ord("A")],
                "llm answer": ans
            }
        )

        # os.remove(new_audio_path)
        os.remove(new_video_path)
        # os.remove(new_combined_path)
    
    print(result)
    return result, records