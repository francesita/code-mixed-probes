import pickle
import time
import json
import networkx as nx
import pandas as pd


# Get the tweet ids for our examples
def get_tweet_ids(cs_conllu):
    tweet_ids = []
    for line in cs_conllu:
        if line.startswith("#"):
            line = line.split()
            if line[1] == 'sent':
                print(line)
                tweet_ids.append(line[3])
    return tweet_ids


# get the data we will use
cs_conllu = open(
    'cs_data_transformed_to_conllu',
    'r', encoding='utf-8')
all_tweet_ids = get_tweet_ids(cs_conllu)


# test_list = prediction_pickle[-2]['edges']
def find_usable_examples(preds_pickle):
    usable = 0
    tweet_ids = []
    for i, item in enumerate(preds_pickle):
        if 0 < len(item['edges']) <= 10:  # then do > 10 <= 12, then do 12 > and <= 15
            usable += 1
            tweet_ids.append({'tweet_id': all_tweet_ids[i], 'tweet_index': i})

    tweet_ids = pd.DataFrame(tweet_ids)

    return usable, tweet_ids


def get_usable_examples(tweet_ids, pickle_file1, pickle_file2):
    examples1 = []
    examples2 = []

    for id in tweet_ids.tweet_index:
        examples1.append(pickle_file1[id])
        examples2.append(pickle_file2[id])  # called en_preds but its for all preds. I change up in the file path area
    # here I make sure that I choose the right example from en or es files when a CS example chosen

    return examples1, examples2


def save_json(dic, filename, jsonspath='/path_to_dependencies/filename.pkl'):
    file = open(jsonspath + filename + ".json", 'w')
    json.dump(dic, file, ensure_ascii=True)


def get_edit_distance(examples_1, examples_2, tweet_ids, json_path, filename, k=0):
    """
    :param k: used to help us pick up where we left off if program crashed at any point, important to keep out json name convention
    :param other_list_dics:
    :param cs_list_dics: a list containing dictionaries with text, pos, and edges of cs
    :param list_dics: a list containing dictionaries with text, pos, and edges of the other lang
    :return: list containing dictionaries with all of the above and edit distances
    """
    # list containing dictionary with distances
    distances_dics = []
    # start time
    st = time.time()

    # for k in range(len(cs_list_dics)):
    while k < len(examples_1):
        print(f"Working on example {k}")
        G1 = nx.Graph()
        print(examples_2[k], 'here')
        G1.add_edges_from(examples_1[k]['edges'])
        G2 = nx.Graph()
        G2.add_edges_from(examples_2[k]['edges'])
        distance = [v for v in nx.optimize_graph_edit_distance(G1, G2)][0]
        distances_dics.append({'tweet_id': tweet_ids.tweet_id[k],
                               'example_1_sentence': examples_1[k]['text'],
                               'example_2_sentence': examples_2[k]['text'],
                               'distance': distance
                               })

        distances_df = pd.DataFrame(distances_dics)
        distances_df.to_csv(f'/path_to_dependencies/filename.pkl'
                            f'/{filename}.csv')

        simgnn_dic = {"graph_1": [[int(e[0]), int(e[1])] for e in G1.edges],
                      "graph_2": [[int(e[0]), int(e[1])] for e in G2.edges],
                      "labels_1": list(G1.nodes),
                      "labels_2": list(G2.nodes),
                      "ged": int(distance)
                      }
        k += 1  # increase k by one

        save_json(simgnn_dic, str(k), jsonspath=json_path)

    # end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    return distances_dics




preds_path_1 = '/path_to_dependencies/filename.pkl'
preds_path_2 = '/path_to_dependencies/filename.pkl'
preds_path_3 = '/path_to_dependencies/filename.pkl'

preds_pickle_1 = pickle.load(open(preds_path_1, 'rb'))
preds_pickle_2 = pickle.load(open(preds_path_2, 'rb'))
preds_pickle_3 = pickle.load(open(preds_path_3, 'rb'))




# read tweet_ids csv
#
ids = pd.read_csv('match_tweet_ids.csv')


#### synthetic calls
preds_path_6 = '/path_to_dependencies/filename.pkl'
preds_path_4 = '/path_to_dependencies/filename.pkl'
preds_path_5 = '/path_to_dependencies/filename.pkl'

preds_pickle_6 = pickle.load(open(preds_path_6, 'rb'))
preds_pickle_4 = pickle.load(open(preds_path_4, 'rb'))
preds_pickle_5 = pickle.load(open(preds_path_5, 'rb'))

#####################   RANDOME
# synthetic examples vs es
examples1, examples2 = get_usable_examples(ids, preds_pickle_6, preds_pickle_2)
# save the distances
distances = get_edit_distance(examples1, examples2, ids, json_path='/path_to_dependencies/',filename='dependencies', k =54)
