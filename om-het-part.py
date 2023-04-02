import math
import os
import json
import numpy as np
import random

random.seed(42)
SPLIT = 1
PARTITION = "B"

IMAGES_FINAL = "leftImg8bit"
TARGET_FINAL = "gtFine_labelIds"

MAX_SAMPLE_PER_CLIENT = 20

ROOT_DIR = "./data/Cityscapes/"

if PARTITION == "A":
    # train A
    with open(os.path.join(ROOT_DIR, "train_A.txt"), "r") as f:
        lines = f.readlines()
        images = [
            (
                l.strip().split("@")[0],
                l.strip().split("@")[1],
            )
            for l in lines
        ]
if PARTITION == "B":
    # train B
    with open(os.path.join(ROOT_DIR, "train.txt"), "r") as f:
        lines = f.readlines()
        images = [
            (
                l.strip().split("/")[1],
                l.strip().split("/")[1].replace(IMAGES_FINAL, TARGET_FINAL),
            )
            for l in lines
        ]


city_dic = {}
for i in images:
    city_name = i[0].split("_")[0]
    if city_name not in city_dic:
        city_dic[city_name] = []
    city_dic[city_name].append(i)

if SPLIT == 1:
    # uniform
    # every client has images from different cityes
    n_sample = len(images)
    n_client_per_city = math.ceil(n_sample / MAX_SAMPLE_PER_CLIENT)
    city_enum = list(enumerate(city_dic.keys()))
    choices = [k for k in city_dic.keys()]
    weights = [len(city_dic[c]) for c in choices]
    client_dict = {}
    for i in range(n_client_per_city):
        client_dict[i] = []
        for _ in range(MAX_SAMPLE_PER_CLIENT):
            choices = [k for k in city_dic.keys()]
            weights = [len(city_dic[c]) for c in choices]
            try:
                c = random.choices(choices, weights=weights, k=1)[0]
            except:
                break
            img, lable = city_dic[c].pop()
            client_dict[i].append((img, lable))
            if len(city_dic[c]) == 0:
                city_dic.pop(c)

    with open(os.path.join(ROOT_DIR, f"uniform{PARTITION}.json"), "w") as outfile:
        json.dump(client_dict, outfile, indent=4)


if SPLIT == 2:
    # heterogeneous
    # every client has images from only one city
    client_dict = {}
    tot_clients = 0
    for city in city_dic.keys():
        n_samples_per_city = len(city_dic[city])
        n_client_per_city = math.ceil(n_samples_per_city / MAX_SAMPLE_PER_CLIENT)
        avg = len(city_dic[city]) // n_client_per_city

        for i in range(tot_clients, tot_clients + n_client_per_city):
            client_dict[i] = []
            for _ in range(avg):
                img, lbl = city_dic[city].pop()
                client_dict[i].append((img, lbl))
            tot_clients += 1
        if len(city_dic[city]) > 0:
            for img, lbl in city_dic[city]:
                client_dict[i].append((img, lbl))
    with open(os.path.join(ROOT_DIR, f"heterogeneuos{PARTITION}.json"), "w") as outfile:
        json.dump(client_dict, outfile, indent=4)
