# InterHub: A Naturalistic Trajectory Dataset with Dense Interaction for Autonomous Driving

## Overview
This repo provides the source code and data for the following paper:

InterHub: A Naturalistic Trajectory Dataset with Dense Interaction for Autonomous Driving

We provide mainly three tools to help users navigate **InterHub**:

1. **1_Data_unify.py**  
   Provides a data interface to convert various data resources into a unified format that works seamlessly with the interaction event extraction process.

2. **2_Interaction_extract.py**  
   Extracts interactive segments from the unified driving records, following the criterion detailed in our [Paper](#citation).

3. **3_Case_visualize.py**  
    Creates GIFs to visualize the interaction cases in **InterHub**, showcasing typical interaction scenarios.


## Quick Start

### Environment Setup

Please make sure the following prerequisites are satisfied. We recommend using conda to manage the python environment.

* Create a conda environment and activate it:
```
conda create --name interhub python=3.8
conda activate interhub
```

* Upgrade pip to the latest version:
```
python -m pip install --upgrade pip
```

* (Optional) Try to change the pip source for faster installation if run into network issues:
```
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

* Install the required packages:
```
pip install -r requirements.txt
```

### Walk through InterHub with a mini dataset
* We provide a subset of the original *interaction_multi* dataset in `data/0_origin_datasets/interaction_multi` for a quick try.
Run the following command to unify the data in the subset:
```
python 0_data_unify.py --desired_data interaction_multi --cache_location data/1_unified_cache/interaction_multi --data_dirs data/0_origin_datasets/interaction_multi
```

* Then, you can run the following command to extract the interaction events from the subset:
```
python 1_Interaction_extract.py --desired_data interaction_multi --cache_location data/1_unified_cache/interaction_multi --save_path data/2_extracted_results
```

* Finally, you can run the following command to visualize the interaction events:
```
python 2_case_visualize.py --cache_location data/1_unified_cache/interaction_multi --interaction_idx_info data/2_extracted_results/results.csv  --top_n 3
```
see `figs/case` for the visualization results.


## Full Working Flows with InterHub

You can use the cache file of the interaction event data that has 
been formatted uniformly by us directly on [this webpage](lianjie), without having to download the original dataset. 

( If you need to extract data from the origin datasets, you will 
need to download the raw datasets. [dataset.md](dataset.md) 
provide more details. )


### 1. Data Unify

**1.1 For those who want to explore the ready-for-use interaction data in the InterHub dataset**, the interaction events in **INTERACTION, nuPlan, Waymo, lyft** are already extracted, unified and processed, one can download and directly use them. Download the data from [InterHub](https://lianjie.link/interhub), unzip it, and put it in the `data/1_unified_cache` folder and move on to step [2. Interaction Event Extract](#2-interaction-event-extract).


**1.2 For those who want to work from scratch or extract interaction events from the data resource other than those in the InterHub**, the origin datasets is necessary in this case. Turn to [Datasets](#datasets) for details to build the needed datasets structure. It is important to preprocess the dataset to form a data cache if you use the initial datasets or other datasets that are not in the list that we have processed.

For the dataset **INTERACTION, nuPlan, Waymo, lyft**, `1_Data_unify.py` provides scripts for preprocessing the raw data into unified data cache that works with the following steps. The project [trajdata](https://github.com/NVlabs/trajdata?tab=readme-ov-file#data-preprocessing-optional) is applied in this step. Replace the following arguments according to the dataset you want to process:

- **desired_data**: A list specifying the datasets to be processed, e.g., `["interaction_multi"]`. See support list in [dataset.md](dataset.md).

- **load_path**: the path where the corresponding raw data is stored, e.g., `'data/0_origin_datasets/interaction_multi'`.

- **cache_location**: The path where the generated cache will be stored. Note that this requires a large amount of storage space, so please ensure you have enough memory, e.g., `'data/1_unified_cache/interaction_multi'`.


## 2. Interaction Event Extract

```commandline
python 2_Interaction_extract.py \
--desired_data dataset_name \
--cache_location path/to/your/dataset/cache \
--save_path path/to/save/your/result\
--num_workers=8 \
--timerange=5
```

Replace `cache_location`, and `save_path` with your own paths. For details, refer to the parameter settings in the [Get trajdata cache](#get-trajdata-cache) section. By default, a subset of `interaction_multi` dataset is read from the `data/1_unified_cache` folder.


## 3. Visualize  

### Case_visualize
Run `2_case_visualize.py` to plot the interaction segments and generate GIFs.

### Paper_plot
Run `3_paper_plot.py` to plot the results in the paper using the metadata of the interaction events in the full InterHub dataset.


## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.



