
# Dataset

## Prerequisites for handling the original datasets in InterHub

We provide the unified and processed interaction event extracted from multiple public datasets in the full [Interhub dataset](https://lianjie.link/interhub) which are ready to use. But if you want to extract the interaction events from the original datasets, you will need to install the following devkits and/or package dependencies:
```
# For Lyft
pip install "trajdata[lyft]"

# For Waymo
pip install "trajdata[waymo]"

# For INTERACTION
pip install "trajdata[interaction]"
```

* If you need to use nuPlan, you will need to install the 
[nuPlan devkit](https://nuplan-devkit.readthedocs.io/en/latest/installation.html) 
separately.

This is the `Installation` step of trajdata, and for more 
information, please see <https://github.com/NVlabs/trajdata?tab=readme-ov-file#installation>


## Download the raw datasets
Download the raw datasets (Waymo, nuPlan, Lyft Level 5, INTERACTION, 
etc.) in case you do not already have them. **If you prefer to 
use our processed data directly, you can skip this step.** However, 
if you have other requirements and need to extract data differently, 
you will need to download these datasets.


* **Waymo Open Motion Dataste:** Download the v1.1 as per [the instructions on the dataset website](https://waymo.com/open/licensing/?continue=%2Fopen%2Fdownload%2F).
Structure the dataset as this:

```
/path/to/waymo/
            ├── training/
            |   ├── training.tfrecord-00000-of-01000
            |   └── ...
            ├── validation/
            │   ├── validation.tfrecord-00000-of-00150
            |   └── ...
            └── testing/
                ├── testing.tfrecord-00000-of-00150
                └── ...
```

* **nuPLan:** Download the v1.1 as per [the instructions in the devkit documentation](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html).
Structure the dataset as this:

```
/path/to/nuPlan/
            └── dataset
                ├── maps
                │   ├── nuplan-maps-v1.0.json
                │   ├── sg-one-north
                │   │   └── 9.17.1964
                │   │       └── map.gpkg
                │   ├── us-ma-boston
                │   │   └── 9.12.1817
                │   │       └── map.gpkg
                │   ├── us-nv-las-vegas-strip
                │   │   └── 9.15.1915
                │   │       ├── drivable_area.npy.npz
                │   │       ├── Intensity.npy.npz
                │   │       └── map.gpkg
                │   └── us-pa-pittsburgh-hazelwood
                │       └── 9.17.1937
                │           └── map.gpkg
                └── nuplan-v1.1
                    ├── mini
                    │   ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
                    │   └── ...
                    └── trainval
                        ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
                        └── ...
```

* **INTERACTION:** Download it as per [the instructions on the dataset website](http://interaction-dataset.com/). 
Structure the dataset as this:

```
/path/to/interaction_single/
            ├── maps/
            │   ├── DR_CHN_Merging_ZS0.osm
            |   ├── DR_CHN_Merging_ZS0.osm_xy
            |   └── ...
            ├── test_conditional-single-agent/
            │   ├── DR_CHN_Merging_ZS0_obs.csv
            |   ├── DR_CHN_Merging_ZS2_obs.csv
            |   └── ...
            └── test_single-agent/
            │   ├── DR_CHN_Merging_ZS0_obs.csv
            |   ├── DR_CHN_Merging_ZS2_obs.csv
            |   └── ...
            └── train/
            │   ├── DR_CHN_Merging_ZS0_train.csv
            |   ├── DR_CHN_Merging_ZS2_train.csv
            |   └── ...
            └── val/
                ├── DR_CHN_Merging_ZS0_val.csv
                ├── DR_CHN_Merging_ZS2_val.csv
                └── ...
```
```
/path/to/interaction_multi/
            ├── maps/
            │   ├── DR_CHN_Merging_ZS0.osm
            |   ├── DR_CHN_Merging_ZS0.osm_xy
            |   └── ...
            ├── test_conditional-multi-agent/
            │   ├── DR_CHN_Merging_ZS0_obs.csv
            |   ├── DR_CHN_Merging_ZS2_obs.csv
            |   └── ...
            └── test_multi-agent/
            │   ├── DR_CHN_Merging_ZS0_obs.csv
            |   ├── DR_CHN_Merging_ZS2_obs.csv
            |   └── ...
            └── train/
            │   ├── DR_CHN_Merging_ZS0_train.csv
            |   ├── DR_CHN_Merging_ZS2_train.csv
            |   └── ...
            └── val/
                ├── DR_CHN_Merging_ZS0_val.csv
                ├── DR_CHN_Merging_ZS2_val.csv
                └── ...
```
* **Lyft:** Download it as per [the instructions on the dataset website](https://woven-planet.github.io/l5kit/dataset.html). 
Structure the dataset as this:

```
/path/to/lyft/
            ├── LICENSE
            ├── aerial_map
            ├── feedback.txt
            ├── meta.json
            ├── scenes/
            │   ├── sample.zarr
            |   ├── train.zarr
            |   └── ...
            └── semantic_map/
                └── semantic_map.pb
```
