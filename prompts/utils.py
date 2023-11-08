import csv
import os
from hashlib import md5


def read_csv_to_dict_list(file_path):
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        list_of_dicts = [row for row in reader]
        return list_of_dicts


def split_list_with_key(lst, dict_key):
    result = {}
    for row in lst:
        if row.get(dict_key) not in result:
            result[row.get(dict_key)] = []
        result[row.get(dict_key)].append(row)
    return result


def read_csv_to_type_dict(file_path, type_key):
    lst = read_csv_to_dict_list(file_path=file_path)
    return split_list_with_key(lst=lst, dict_key=type_key)


def md5_str(str):
    return md5(str.encode("utf8")).hexdigest()


current_dir = os.path.dirname(__file__)


class PromtsContainer(object):
    def __init__(self) -> None:
        prompts_path = os.path.join(current_dir, "prompts_en.csv")
        self.data = read_csv_to_type_dict(prompts_path, "type")
        self.summary_dict = {
            md5_str(row.get("summary")): row.get("prompt")
            for chunk in self.data.values()
            for row in chunk
        }

    def get_prompts_tab_dict(self):
        return self.data

    def get_prompt_by_summary(self, summary):
        return self.summary_dict.get(md5_str(summary), summary)
