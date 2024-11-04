# InterHub: A Dense Driving Interaction Dataset with Standardized Scene Description

## Overview
This repo provides the source code and data for the following paper:\

^^^^ 

<!-- The process contains the following three steps:\

1. **Get trajdata cache** to form the unified dataset.
2. **Extract interactive driving events** to obtain the interactive segments index information. (This step can be ignored, since the index information of Waymo, nuPlan, Lyft, INTERACTION is provided in <https://dddddddddd>)
3. **Retrieve interactive segments** and former usage.
 -->

We provide mainly three tools to help users navigate **InterHub**:

1. **1_Data_unify.py**  
   Provides a data interface to convert various data resources into a unified format that works seamlessly with the interaction event extraction process.

2. **2_Interaction_extract.py**  
   Extracts interactive segments from the unified driving records, following the criterion detailed in the [Paper](#citation).

3. **3_Case_visualize.py**  
    Creates GIFs to visualize the interaction cases in **InterHub**, showcasing typical interaction scenarios.


4. **4_Paper_plot.py**  
    Generates plots of statistical data featured in the paper, visualizing key metrics and findings from the analyzed datasets.

All programs can be run on the `Dataset/InterHub_mini` dataset we have extracted.



## Environment and Dataset

### Environment Setup

All the data are provided to index from trajdata cache, if you want to use the data provided by us or you want to run our source code, please make sure the following prerequisites are satisfied.
* Create a conda environment:
```
conda create --name interhub python=3.9
```

* Activate the environment:
```
conda activate interhub
```

### change the pip source to tsinghua for China users
```
python -m pip install --upgrade pip
# pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

* Install the required packages:
```
pip install -r requirements.txt
```

* The following will install the respective devkits and/or package dependencies:
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


### Datasets

You can use the cache file of the interaction event data that has 
been formatted uniformly by us directly on [this webpage](lianjie), without 
having to download the original dataset. 

( If you need to extract data from the origin datasets, you will 
need to download the raw datasets. [Dataset.md](dataset.md) 
provide more details. )



## 1. Data Unify
**The origin datasets is necessary for this step, turn to [Datasets](#datasets) for details to 
build the needed datasets structure. If you want to use the data we 
processed directly, please skip to steps [2. Interaction Event Extract](#2-interaction-event-extract)**

It is important to preprocess the dataset to form a data cache if 
you use the initial datasets or other datasets that are not in the 
list that we have processed.

For the dataset in **INTERACTION, nuPlan, Waymo, lyft**, you can 
use`trajdata` for preprocessing. Take a look at the `1_Data_unify.py` 
script for an example of how to do this.

Replace the following three parameters:

- **desired_data**: A list specifying the datasets to be processed, e.g., `["lyft_train_full"]`.
- **cache_location**: The path where the generated cache will be stored. Note that this requires a large amount of storage space, so please ensure you have enough memory, e.g., `'Dataset/lyft'`.
- **data_dirs**: A d7ictionary mapping each element in `desired_data` to the path where the corresponding raw data is stored, e.g., 
  ```python
  {
      "lyft_train_full": 'dataset/lyft/scenes/train_full.zarr'
  }

This is the `Data Preprocessing` step of `trajdata`. For more information, please click [here](https://github.com/NVlabs/trajdata?tab=readme-ov-file#data-preprocessing-optional).

## 2. Interaction Event Extract

```commandline
python 2_Interaction_extract.py \
--desired_data dataset_name \
--cache_location path/to/your/dataset/cache \
--save_path path/to/save/your/result\
--num_workers=8 \
--timerange=5
```

Replace `cache_location`, and `save_path` with your own paths. For details, refer 
to the parameter settings in the [Get trajdata cache](#get-trajdata-cache) section. 
By default, the `InterHub_mini` dataset is read from the `Dataset` 
folder.


## 3. Visualize  

### Case_visualize
Run `plot_figure.py` to plot the interaction segments, and the gif will be provided.

### Paper_plot
Run `contextVAE/process_trajdata.py` to preprocess the dataset to meet the data format requirements of [ContextVAE](https://github.com/xupei0610/ContextVAE.git).

Then, test the model on the preprocessed data.  **_( More Details)_**



## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.0.



