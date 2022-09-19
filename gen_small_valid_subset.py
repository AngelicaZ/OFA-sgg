from data.mm_data.sg_raw import GQASceneDataset, load_json, np_load
import random
import json

'''
Generate smaller validation subset to train faster
'''
def main():
    new_json = dict()

    valid_dataset_path = '/data/c/zhuowan/gqa/data/sceneGraphs/val_sceneGraphs.json'
    new_json_path = '/home/chenyu/scene_graph_generation/OFA/dataset/sgg_data/GQA/val_sg_small_subset.json'

    scenegraphs_json = load_json(valid_dataset_path)
    keys = list(scenegraphs_json.keys())
    total_img_num = len(keys)
    print("total_img_num: ", total_img_num)

    subset_len = 100
    # for i in range(total_img_num // subset_len):
    for i in range(subset_len):
        new_json[keys[i]] = scenegraphs_json[keys[i]]
    
    json_object = json.dumps(new_json, indent=4)

    with open(new_json_path, 'w') as f:
        f.write(json_object)



if __name__ == "__main__":
    main()