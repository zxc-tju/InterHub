
# Dataset

## Dataset Information

The **metadata file** provided in InterHub is stored in csv format, and the columns and descriptions are as follows:

| Column               | Data Type | Information                                                                                                                      |
|----------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------|
| `dataset`            | `str`     | ID of the dataset: One of `{'nuplan_train', 'waymo_train', 'interaction_single', 'interaction_multi', 'lyft_train_full'}`.       |
| `folder`               | `str`     | Parent folder of the cache file. **For each scenario, please read the corresponding cache file from the Parent folder.**                                                                                              |
| `scenario_idx`       | `int64`   | Index of the scenario.                                                                                                          |
| `track_id`           | `str`     | Agent ID of vehicles involved in the interaction: Separated by semicolons (`;`).                                               |
| `start`              | `float`   | Start time of the interaction segment.                                                                                          |
| `end`                | `float`   | End time of the interaction segment.                                                                                            |
| `intensity`          | `float`   | Intensity of the interaction segment.                                                                                           |
| `PET`                | `float`   | PET (Post-Encroachment Time) of the interaction segment.                                                                        |
| `two/multi`      | `str`     | Number of vehicles involved in the interaction: One of `{'two', 'multi'}`.                                                     |
| `vehicle_type`       | `str`     | The type of each vehicle involved in the interaction (e.g., `['HV', 'AV']`, where `HV` stands for human-driven vehicles and `AV` stands for autonomous vehicles). |
| `AV_included`        | `str`     | Whether autonomous vehicles are involved in the interaction: One of `{'all_HV', 'AV'}`.                                        |
| `key_agents`         | `str`     | The IDs of the two key vehicles in the interaction: Separated by semicolons (`;`).                                             |
| `pre_int_i`          | `int64`   | Time step for valid data selected **before** the intersection of the first vehicle in `key_agents`: Default is `50`. A value less than `50` indicates insufficient valid trajectory points. |
| `post_int_i`         | `int64`   | Time step for valid data selected **after** the intersection of the first vehicle in `key_agents`: Default is `50`. A value less than `50` indicates insufficient valid trajectory points. |
| `pre_int_j`          | `int64`   | Time step for valid data selected **before** the intersection of the second vehicle in `key_agents`: Default is `50`. A value less than `50` indicates insufficient valid trajectory points. |
| `post_int_j`         | `int64`   | Time step for valid data selected **after** the intersection of the second vehicle in `key_agents`: Default is `50`. A value less than `50` indicates insufficient valid trajectory points. |
| `path_category`      | `str`     | The trajectory relationship label between `key_agents`: One of `{'CP', 'MP', 'HO', 'F'}`.                                           |
| `path_relation`      | `str`     | The driving direction relationship label **before** and **after** the intersection (e.g., `P-M`, `C-O`). <br> - `P-M`: The two agents were running parallel (`P`) before the intersection and merged (`M`) after the intersection. <br> - `C-O`: The two agents were running crossed (`C`) before the intersection and opposite (`O`) after the intersection. |
| `turn_label`         | `str`     | The turning direction of the two vehicles: Recorded in the `td_i-td_j` format, where `td_i` and `td_j` represent the turning directions, each being one of: <br> - `S` (straight) <br> - `L` (left turn) <br> - `R` (right turn) <br> - `U` (U-turn). |
| `priority_label`     | `str`     | The ID of the vehicle with right of priority among the `key_agents`.                                                           |

## Prerequisites for handling the original datasets in InterHub

We provide the unified and processed interaction event extracted from multiple public datasets in the full [Interhub dataset](https://figshare.com/articles/dataset/_b_InterHub_A_Naturalistic_Trajectory_Dataset_with_Dense_Interaction_for_Autonomous_Driving_b_/27899754) which are ready to use. But if you want to extract the interaction events from the original datasets, you will need to install the following devkits and/or package dependencies:
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
