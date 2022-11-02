import json
import random
from random import shuffle
from argparse import ArgumentParser

random.seed(8)

def preprocess_sherlock(mini_flag=True):
    data = []
    val_data = []

    num_val_and_test_urls = 70
    num_test_urls = 50

    base_directory = "$SCRATCH/multimodal/data/sherlock/"
    if not mini_flag:
        new_train_file_name = base_directory + "new_sherlock_train.json"
        new_test_file_name = base_directory + "new_sherlock_test.json"
        new_val_file_name = base_directory+ "new_sherlock_val.json"
        val_url_list_file_name = base_directory + "val_url_list.json"
        test_url_list_file_name = base_directory + "test_url_list.json"
    else:
        print("Creating MINI dataset")
        len_train_data = 20
        num_val_and_test_urls = 5
        num_test_urls = 2
        new_train_file_name = base_directory +"mini/new_sherlock_train.json"
        new_test_file_name = base_directory + "mini/new_sherlock_test.json"
        new_val_file_name = base_directory + "mini/new_sherlock_val.json"
        val_url_list_file_name = base_directory + "mini/val_url_list.json"
        test_url_list_file_name = base_directory + "mini/test_url_list.json"

    with open('/home/nm3571/multimodal/sherlock/sherlock_train_v1_1.json', 'r') as f:
    data = json.load(f)
    with open('/home/nm3571/multimodal/sherlock/sherlock_val_with_split_idxs_v1_1.json', 'r') as f:
    val_data = json.load(f)

    print("-------BEFORE PROCESSING--------")
    print("Length of training data", len(data))
    print("Length of validation data", len(val_data))
    print("Total", len(data)+len(val_data))

    url_dict = dict()
    for i, item in enumerate(val_data):
    url = item["inputs"]["image"]["url"]
    if url not in url_dict:
        url_dict[url] = []
    url_dict[url].append(item)

    random_val_and_test_urls = set(random.sample(list(url_dict), num_val_and_test_urls))
    random_test_urls = set(random.sample(list(random_val_and_test_urls), num_test_urls))

    test_json_list = []

    for url in random_test_urls:
        test_json_list.extend(url_dict[url])

    val_json_list = []
    random_val_urls = []
    for url in random_val_and_test_urls:
    if url not in random_test_urls:
        random_val_urls.append(url)
        val_json_list.extend(url_dict[url])

    for url in url_dict:
    if url not in random_val_and_test_urls:
        data.extend(url_dict[url])

    if mini_flag:
        data = data[:len_train_data]

    shuffle(val_json_list)
    shuffle(test_json_list)
    shuffle(data)

    print("-------AFTER PROCESSING--------")
    print("Length of train data", len(data))
    print("Length of val data", len(val_json_list))
    print("Length of test data", len(test_json_list))
    print("Total", len(data)+len(val_json_list)+len(test_json_list))

    with open(new_train_file_name, 'w') as f:
        json.dump(data, f)


    with open(new_val_file_name, 'w') as f:
    json.dump(val_json_list, f)

    with open(new_test_file_name, 'w') as f:
    json.dump(test_json_list, f)

    with open(test_url_list_file_name, 'w') as f:
    json.dump(list(random_test_urls), f)

    with open(val_url_list_file_name, 'w') as f:
    json.dump(list(random_val_urls), f)

    print("Created new train, val and test json files.")

parser = ArgumentParser()
parser.add_argument('--mini', type=bool, action='store_true') 

args = parser.parse_args()

preprocess_sherlock(args.mini)

