import json
import os

# list all file
file_list = os.listdir('summary')

train_story_name = []
test_story_name = []
val_story_name = []

train_path = '../data/FairytaleQA_Dataset/FairytaleQA_Dataset_Sentence_Split/split_for_training/train'
test_path = '../data/FairytaleQA_Dataset/FairytaleQA_Dataset_Sentence_Split/split_for_training/test'
val_path = '../data/FairytaleQA_Dataset/FairytaleQA_Dataset_Sentence_Split/split_for_training/val'

train_story_list = os.listdir(train_path)
test_story_list = os.listdir(test_path)
val_story_list = os.listdir(val_path)

for story in train_story_list:
    if story[-10:] == '-story.csv':
        train_story_name.append(story.replace('-story.csv', ''))

for story in test_story_list:
    if story[-10:] == '-story.csv':
        test_story_name.append(story.replace('-story.csv', ''))

for story in val_story_list:
    if story[-10:] == '-story.csv':
        val_story_name.append(story.replace('-story.csv', ''))

print(len(train_story_name), len(test_story_name), len(val_story_name))

train_summary_dataset = {}
test_summary_dataset = {}
val_summary_dataset = {}

for train_story in train_story_name:
    with open(f'summary/{train_story}.json') as f:
        json_data = json.load(f)
        s = json_data[0]['summary']
        train_summary_dataset[train_story] = s

for test_story in test_story_name:
    with open(f'summary/{test_story}.json') as f:
        json_data = json.load(f)
        s = json_data[0]['summary']
        test_summary_dataset[test_story] = s
    
for val_story in val_story_name:
    with open(f'summary/{val_story}.json') as f:
        json_data = json.load(f)
        s = json_data[0]['summary']
        val_summary_dataset[val_story] = s

# check folder exist
if not os.path.exists('data'):
    os.mkdir('data')

with open('data/summary_train.json', 'w') as f:
    json.dump(train_summary_dataset, f, indent=4, ensure_ascii=False)

with open('data/summary_test.json', 'w') as f:
    json.dump(test_summary_dataset, f, indent=4, ensure_ascii=False)

with open('data/summary_val.json', 'w') as f:
    json.dump(val_summary_dataset, f, indent=4, ensure_ascii=False)
