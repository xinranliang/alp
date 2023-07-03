This folder contains instructions to prepare pretraining and downstream evaluation datasets.

## Preparing training dataset

Download the Gibson dataset using the instructions [here](https://github.com/facebookresearch/habitat-lab#scenes-datasets) (download the 11GB file `gibson_habitat_trainval.zip`)

Move the Gibson scene dataset or create a symlink at `data/scene_datasets/gibson_semantics`.

Download the tiny partition of the 3DSceneGraph dataset which contains the semantic information for the Gibson dataset.

The semantic data will need to be converted before it can be used within Habitat, follow instructions to use `gen_gibson_semantics.sh` script from the [habitat-sim](https://github.com/facebookresearch/habitat-sim#datasets).

Download the tiny partition (`gibson_tiny.tar.gz`) of the gibson dataset and extract it into the `data/scene_datasets/gibson_tiny` folder.

Run script to generate semantic annotations:
```sh
bash habitat-sim/datatool/tools/gen_gibson_semantics.sh data/scene_datasets/3DSceneGraphTiny/automated_graph data/scene_datasets/gibson_tiny data/scene_datasets/gibson_semantics
```

## Setting up training dataset

The code requires setting up following formats in the `data` folder:
```
alp/
  data/
    scene_datasets/
      gibson_semantics/
        # for non-semantic scenes
        Airport.glb
        Airport.navmesh
        # for semantic scenes
        Allensville_semantic.ply
        Allensville.glb
        Allensville.ids
        Allensville.navmesh
        Allensville.scn
        ...

```

To test training datasets are downloaded in correct formats, run `python data/example_test.py` should print out set of objects in Allensville scene in Gibson dataset, mappings from category_id to instance_id, and mappings from instance_id to category_id. You could use scripts `python data/example_test.py --scene [SCENE_NAME]` to test in other Gibson scenes.

## Setting up evaluation dataset

To collect evaluation dataset for downstream perception tasks, run `python data/collect_eval.py --mode [holdout, test]` to collect evaluation images for Train Split and Test Split in Gibson dataset. Evaluation images should be saved in `data/eval/holdout/` for Train Split and `data/eval/test/` for Test Split.
