import os
import cv2
import pickle
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pycocotools
import torch

import habitat_sim
import habitat
from habitat_baselines.common.sim_settings import default_sim_settings, make_cfg
from habitat_baselines.common.constants import scenes, master_scene_dir, coco_categories, coco_categories_mapping

import detectron2
from detectron2.structures import BoxMode


def make_custom_cfg(scene_str):
    default_sim_settings['scene'] = master_scene_dir + scene_str + '.glb'
    default_sim_settings['width'] = 256
    default_sim_settings['height'] = 256
    default_sim_settings['sensor_height'] = 0.8
    default_sim_settings['semantic_sensor'] = True
    default_sim_settings['depth_sensor'] = True
    default_sim_settings['seed'] = 54321 # training mode
    # default_sim_settings['seed'] = 12345 # test mode

    cfg = make_cfg(default_sim_settings, torch.cuda.current_device())
    return cfg


def initialize_scene_agent(cfg):
    sim = habitat_sim.Simulator(cfg)
    random.seed(default_sim_settings["seed"])
    sim.seed(default_sim_settings["seed"])

    # initialize the agent at a random start state
    agent = sim.initialize_agent(default_sim_settings["default_agent"])
    start_state = agent.get_state()

    # force starting position on first floor (try 100 samples)
    num_start_tries = 0
    while num_start_tries < 100:
        start_state.position = sim.pathfinder.get_random_navigable_point()
        num_start_tries += 1
    agent.set_state(start_state)
    return sim, agent


def print_scene_objects(sim):
    scene = sim.semantic_scene

    # key: category_id as specified in coco_categories
    #      "15" is for background
    # values: list of instance_id falling into this category_id
    category_instance_lists = {}

    # key: instance_id
    # value: category_id as specified in coco_categories
    #        "6" is for background
    instance_category_lists = {}

    for obj in scene.objects:
        if obj is None or obj.category is None:
            continue
        print(
                f"Object id:{obj.id}, category:{obj.category.name()},"
                f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            )
        obj_class = obj.category.name()
        if obj_class in coco_categories.keys():
            cat_id = coco_categories[obj_class]
            obj_id = int(obj.id.split("_")[-1])
            if cat_id not in category_instance_lists:
                category_instance_lists[cat_id] = [obj_id]
            else:
                category_instance_lists[cat_id].append(obj_id)
            if obj_id not in instance_category_lists:
                instance_category_lists[obj_id] = cat_id
    
    print(category_instance_lists)
    print(instance_category_lists)
    # try to compute bounding box

    # input("Press Enter to continue...")
    return category_instance_lists, instance_category_lists


def area_filter(mask, bounding_box, img_height, img_width, size_tol=0.05):
    """
    Function to filter out masks that contain sparse instances
    for example:
        0 0 0 0 0 0
        1 0 0 0 0 0
        1 0 0 0 1 0    This is a sparse mask
        0 0 0 0 1 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        1 1 1 1 1 0
        1 1 1 1 1 1    This is not a sparse mask
        0 0 0 1 1 1
        0 0 0 0 0 0
    """
    xmin, ymin, xmax, ymax = bounding_box
    num_positive_pixels = np.sum(mask[ymin:ymax, xmin:xmax])
    num_total_pixels = (xmax - xmin) * (ymax - ymin)
    big_enough = (xmax - xmin) >= size_tol * img_width and (
        ymax - ymin
    ) >= size_tol * img_height
    if big_enough:
        not_sparse = num_positive_pixels / num_total_pixels >= 0.3
    else:
        not_sparse = False
    return not_sparse and big_enough


def save_semantic_observation(color_obs, semantic_obs, scene_str, instance_to_category, category_to_instance, total_frames):
    record = {}
    
    color_density = 1 - np.count_nonzero(color_obs) / np.prod(color_obs.shape)
    semantic_density = np.count_nonzero(semantic_obs) / np.prod(semantic_obs.shape)
    category_obs = np.empty_like(semantic_obs)
    if color_density < 0.05 and semantic_density > 0.05 ** 2:
        idx = "%s_%03d" % (scene_str, total_frames)
        height, width = default_sim_settings['height'], default_sim_settings['width']

        # record["file_name"] = img_dir
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []

        for instance_id in list(instance_to_category.keys()):
            px = []
            py = []
            obj_mask = np.empty_like(semantic_obs)

            for x in range(semantic_obs.shape[0]):
                for y in range(semantic_obs.shape[1]):

                    # cat_id of sem_seg[x, y]
                    element = semantic_obs[x][y]

                    if element == instance_id:
                        category = instance_to_category[element]

                        # check consistency
                        list_of_objects = category_to_instance[category]
                        assert element in list_of_objects

                        category_obs[x][y] = category
                        obj_mask[x][y] = 1

                        px.append(y)
                        py.append(x)
                    
                    else:
                        category_obs[x][y] = 6
                        obj_mask[x][y] = 0
            
            assert len(px) == len(py)
            if len(px) > 0 and len(py) > 0:
                obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": pycocotools.mask.encode(np.asarray(obj_mask, order="F").astype('uint8')),
                        "category_id": category,
                        }
                
                filter_obj = area_filter(obj_mask, obj['bbox'], height, width)
                if filter_obj:
                    objs.append(obj)
        
        record["annotations"] = objs


def run_env(scene_str, max_frame, start_frame):
    custom_cfg = make_custom_cfg(scene_str)
    sim, agent = initialize_scene_agent(custom_cfg)
    print("agent_state: position", agent.get_state().position, "rotation", agent.get_state().rotation)

    # get instance_id to category_id mapping
    # get category_id to instance_id mapping
    category_instance_lists, instance_category_lists = print_scene_objects(sim)

    total_frames = start_frame
    action_names = list(
        custom_cfg.agents[
            default_sim_settings["default_agent"]
        ].action_space.keys()
    )

    while total_frames < max_frame:
        action = random.choice(action_names)
        
        observations = sim.step(action)
        agent_state = agent.get_state()
        # print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        color_obs = observations["color_sensor"]
        semantic_obs = observations["semantic_sensor"]
        depth_obs = observations["depth_sensor"]

        save_semantic_observation(color_obs, semantic_obs, scene_str, instance_category_lists, category_instance_lists, total_frames)
        print(total_frames)

        total_frames += 1

    sim.close()


def main(args):
    dataset_dict = []
    print('Start running in enviornment ', args.scene)
    run_env(args.scene, max_frame = 10, start_frame = 0)
    print('Finish running in enviornment ', args.scene)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process random data collection arguments.')
    parser.add_argument('--scene', type=str, default="Allensville", help='to only collect one scene with specified name')

    args = parser.parse_args()
    main(args)