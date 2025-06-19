from datasets import load_dataset
from torch.utils.data import Dataset

class AVQA_Dataset(Dataset):
    def __init__(self, doc, category = None, sub_category = None, task_id = None):
        self.questions = []

        for question in doc:
            category_check, sub_category_check, task_id_check = True, True, True
            if category is not None and category != question['category']:
                category_check = False
            if sub_category is not None and sub_category != question['sub_category']:
                sub_category_check = False
            if task_id is not None and task_id != question['task_id']:
                task_id_check = False
            if category_check and sub_category_check and task_id_check:
                self.questions.append(question)       

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        cache_dir = os.getenv("HF_HOME")
        print(cache_dir)
        return self.questions[idx]

def get_dataset(category = None, sub_category = None, task_id = None):
    doc = load_dataset("plnguyen2908/Holistic_AVQA_bench", split="test")
    dataset = AVQA_Dataset(doc, category, sub_category, task_id)
    return dataset