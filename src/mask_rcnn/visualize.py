# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import pickle

# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
# constants
from habitat_baselines.common.constants import scenes, master_scene_dir, coco_categories, coco_categories_mapping

coco_categories_mapping_inverse = {
    0: 56,  # chair
    1: 57,  # couch
    2: 58,  # potted plant
    3: 59,  # bed
    4: 61,  # toilet
    5: 62,  # tv
    # 60: 6,  # dining-table
    # 69: 7,  # oven
    # 71: 8,  # sink
    # 72: 9,  # refrigerator
    # 73: 10,  # book
    # 74: 11,  # clock
    # 75: 12,  # vase
    # 41: 13,  # cup
    # 39: 14,  # bottle
}

coco_categories_objects = ["chair", "couch", "potted plant", "bed", "toilet", "tv"]
master_save_test_dir = "data/eval/test/"
master_save_holdout_dir = "data/eval/holdout/"

def get_all_coco_categories():
    cfg = get_cfg()
    cfg.merge_from_file("configs/mask_rcnn/mask_rcnn_R_50_FPN_3x.yaml")
    object_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    return object_classes

def get_habitat_dicts(scene_str, master_save_dir, max_frames):
    dataset_dicts = []
    for idx in range(max_frames):
        with open(master_save_dir + "%s/dict/%03d.pkl" % (scene_str, idx), 'rb') as f:
            dict_obj = pickle.load(f)
            for single_object in dict_obj['annotations']:
                simple_catid = single_object['category_id']
                single_object['category_id'] = coco_categories_mapping_inverse[simple_catid]
        dataset_dicts.append(dict_obj)
    return dataset_dicts

def main_test():
    coco_objects = get_all_coco_categories()
    for scene_name in scenes['val']:
        DatasetCatalog.register("habitat_" + scene_name, lambda scene_name=scene_name: get_habitat_dicts(scene_name, master_save_dir=master_save_test_dir, max_frames=1000))
        MetadataCatalog.get("habitat_" + scene_name).set(thing_classes=coco_objects)
    
    for scene_name in scenes['val']:
        habitat_metadata = MetadataCatalog.get("haitat_" + scene_name)
        dataset_dicts = get_habitat_dicts(scene_name, master_save_dir=master_save_test_dir, max_frames=1000)
        os.makedirs(os.path.join(master_save_test_dir, scene_name, "label"), exist_ok=True)
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=habitat_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            cv2.imwrite(os.path.join(master_save_test_dir, scene_name, "label", d['image_id'].split('_')[1] + '.png'), out.get_image()[:, :, ::-1])

def main_holdout():
    coco_objects = get_all_coco_categories()
    for scene_name in scenes['train']:
        DatasetCatalog.register("habitat_" + scene_name, lambda scene_name=scene_name: get_habitat_dicts(scene_name, master_save_dir=master_save_holdout_dir, max_frames=200))
        MetadataCatalog.get("habitat_" + scene_name).set(thing_classes=coco_objects)
    
    for scene_name in scenes['train']:
        habitat_metadata = MetadataCatalog.get("haitat_" + scene_name)
        dataset_dicts = get_habitat_dicts(scene_name, master_save_dir=master_save_holdout_dir, max_frames=60)
        os.makedirs(os.path.join(master_save_holdout_dir, scene_name, "label"), exist_ok=True)
        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=habitat_metadata, scale=1)
            out = visualizer.draw_dataset_dict(d)
            cv2.imwrite(os.path.join(master_save_holdout_dir, scene_name, "label", d['image_id'].split('_')[1] + '.png'), out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process groundtruth label visualization.')
    parser.add_argument('--mode', type=str, help='mode: holdout or test')
    args = parser.parse_args()
    
    if args.mode == 'holdout':
        main_holdout()
    elif args.mode == 'test':
        main_test()
    else:
        raise NotImplementedError
