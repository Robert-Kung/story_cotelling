import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from datetime import datetime

# setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

def calc_plot_point_score(x, degree=-0.02, max_score=10, min_score=1):
    return np.exp(degree * x) * (max_score - min_score) + min_score
    # return x

# load all plot points
PATH = 'plot_point'
list_dir = os.listdir(PATH)

plot_point_dataset = {}

# read each file
score_collect_list = []
for json_file in list_dir:
    if json_file.find('.json') == -1:
        continue
    with open(os.path.join(PATH, json_file), 'r', encoding='utf8') as f:
        json_data = json.load(f)
    plot_point_dataset[json_file.replace('.json', '')] = json_data

for story in tqdm(plot_point_dataset.keys()):
    # read each plot point set
    noised_score_plot_point = []
    for plot_points in plot_point_dataset[story]:
        plot_point_num = len(plot_points['plot_points'])
        # random delete plot point
        for delete_num in range(plot_point_num - 1):
            for i in range(delete_num):
                delete_idx_list = []
                while len(delete_idx_list) < delete_num:
                    random_idx = random.randint(0, plot_point_num - 1)
                    if random_idx not in delete_idx_list:
                        delete_idx_list.append(random_idx)

                new_plot_points = copy.deepcopy(plot_points)
                # delete plot point by delete_idx_list
                delete_idx_list.sort(reverse=True)
                for delete_idx in delete_idx_list:
                    # replace plot point
                    new_plot_points['plot_points'].pop(delete_idx)

                # calculate current plot point score
                score = calc_plot_point_score((delete_num/plot_point_num) * 80, degree=-0.02, max_score=10, min_score=1)

                # append to dataset
                new_plot_points['generate_func'] += f'-delete({i + 1})'
                new_plot_points['score'] = round(score, 2)
                new_plot_points['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_plot_points['plot_point_num'] = len(new_plot_points['plot_points'])
                for plot_id in range(len(new_plot_points['plot_points'])):
                    new_plot_points['plot_points'][plot_id]['id'] = plot_id + 1
                noised_score_plot_point.append(new_plot_points)

                score_collect_list.append(score)

        # random replace some plot point from another story
        for replace_num in range(plot_point_num - 1):
            for i in range(replace_num):
                replace_idx_list = []
                while len(replace_idx_list) < replace_num:
                    random_idx = random.randint(0, plot_point_num - 1)
                    if random_idx not in replace_idx_list:
                        replace_idx_list.append(random_idx)

                new_plot_points = copy.deepcopy(plot_points)
                # replace plot point by replace_idx_list
                for replace_idx in replace_idx_list:
                    # random choose another story
                    another_story = random.choice(list(plot_point_dataset.keys()))
                    while another_story == story:
                        another_story = random.choice(list(plot_point_dataset.keys()))
                    # random choose another plot point set
                    another_plot_points = random.choice(plot_point_dataset[another_story])
                    another_plot_points_rdm_idx = random.randint(0, len(another_plot_points['plot_points'])-1)
                    # replace plot point
                    new_plot_points['plot_points'][replace_idx] = another_plot_points['plot_points'][another_plot_points_rdm_idx]

                # calculate current plot point score
                score = calc_plot_point_score((replace_num/plot_point_num) * 100, degree=-0.04, max_score=10, min_score=0)

                # append to dataset
                new_plot_points['generate_func'] += f'-replace({i + 1})'
                new_plot_points['score'] = round(score, 2)
                new_plot_points['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_plot_points['plot_point_num'] = len(new_plot_points['plot_points'])
                for plot_id in range(len(new_plot_points['plot_points'])):
                    new_plot_points['plot_points'][plot_id]['id'] = plot_id + 1
                noised_score_plot_point.append(new_plot_points)
               
                score_collect_list.append(score)

        score_collect_list.append(10)
    plot_point_dataset[story].extend(noised_score_plot_point)

# check folder exist
if not os.path.exists('mixed_plot_point'):
    os.makedirs('mixed_plot_point')

for story in tqdm(plot_point_dataset.keys()):
    random.shuffle(plot_point_dataset[story])
    score_count = {}
    for plot_points in plot_point_dataset[story][:]:
        score = round(plot_points['score'])
        score_count[score] = score_count.get(score, 0) + 1
        if score_count[score] > 9 and score != 10:
            plot_point_dataset[story].remove(plot_points)

    with open(os.path.join('mixed_plot_point', f'{story}.json'), 'w', encoding='utf8') as f:
        json.dump(plot_point_dataset[story], f, ensure_ascii=False, indent=4)

# plt.hist(score_collect_list, density=False, color='blue', cumulative=False, label="Before")
# plt.legend()

# total count score
score_count = {}
score_collect_list = []
for story in plot_point_dataset.keys():
    for plot_points in plot_point_dataset[story]:
        score = round(plot_points['score'])
        score_count[score] = score_count.get(score,0) + 1
        score_collect_list.append(plot_points['score'])
print(score_count)

# plt.hist(score_collect_list, density=False, color='red', cumulative=False, label="After")
# plt.legend()
# plt.xlabel('Score range (0-10)')
# plt.savefig('after.png')