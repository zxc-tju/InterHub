<p align="center">
    <img src="image/logo_interhub.png" alt="Logo" width="500">
    <h1 align="center">A Naturalistic Trajectory Dataset with Dense Interaction for Autonomous Driving</h1>

</p>

<br/>

> [**InterHub: A Naturalistic Trajectory Dataset with Dense Interaction for Autonomous Driving**](https://www.nature.com/articles/s41597-025-05344-7)  <br>
> Published in *Scientific Data*  <br>
> [Xiyan Jiang](https://tops.tongji.edu.cn/info/1161/2143.htm)<sup>1</sup>, [Xiaocong Zhao](https://zxc-tju.github.io/)<sup>1,\*</sup>, [Yiru Liu](https://tops.tongji.edu.cn/info/1131/1810.htm)<sup>1</sup>, [Zirui Li](https://lzrbit.github.io/)<sup>2</sup>, [Peng Hang](https://tops.tongji.edu.cn/info/1031/1383.htm)<sup>1</sup>, Lu Xiong<sup>1</sup>, and [Jian Sun](https://tops.tongji.edu.cn/info/1031/1187.htm)<sup>1,\*</sup>  <br>
> <sup>1</sup> Tongji University, <sup>2</sup> Beijing Institute of Technology
<br>
<sup> * </sup> Correspondance: zhaoxc@tongji.edu.cn, sunjian@tongji.edu.cn

This repo is intended to serve as a starting point for driving-interaction-related research. We provide (a) a publicly accessible dataset [InterHub](https://figshare.com/articles/dataset/_b_InterHub_A_Naturalistic_Trajectory_Dataset_with_Dense_Interaction_for_Autonomous_Driving_b_/27899754) with rich interaction events and (b) tools for interaction extraction.

| ![Teaser GIF 1](image/teaser_1.gif) | ![Teaser GIF 2](image/teaser_2.gif) |
|--------------------------------------|--------------------------------------|
| ![Teaser GIF 3](image/teaser_3.gif) | ![Teaser GIF 4](image/teaser_4.gif) |

<div align="center">
  <a href="https://www.youtube.com/watch?v=OxOpcXYZkEo">
    <img src="image/video_cover.jpg" alt="InterHub Demo Video" width="600">
  </a>
  
</div>

## Roadmap

> Naturalistic driving datasets are reorganized using a unified data interface [trajdata](https://github.com/NVlabs/trajdata?tab=readme-ov-file#data-preprocessing-optional) to provide extensibility and easy access to multiple driving data resources. Then, driving interaction events covering a wide range of interaction archetypes, as well as their combinations, are extracted using the formal method. Rich features of the extracted scenarios, including interaction intensity, AV involvement, and conflict type, are analyzed and annotated to support applications with varied needs regarding driving interaction data.

<div align="center">
<img src="image/roadmap_Interhub.png"/>
</div>

## Overview

### Toolkit
We offer three tools to help users navigate **InterHub**:

* **0_data_unify.py** converts various data resources into a unified format for seamless interaction event extraction.

* **1_interaction_extract.py** extracts interactive segments from unified driving records.

* **2_case_visualize.py** showcases typical interaction scenarios in **InterHub**.

### InterHub

Referencing the method proposed in [Li G, Jiao Y, Calvert S C, et al.](https://doi.org/10.1016/j.trc.2024.104802) (with our approach considering merging scenarios), we classify the interaction into 12 categories based on the driving direction relationship between two key agents before and after the intersection:

<div align="center">
<img src="image/Semantic_Label.png"/>
</div>

In addition to indexing and tracing information about interaction scenarios, we also provide the following interesting labels to facilitate more targeted retrieval and utilization of interaction scenarios.

| Column               | Data Type | Information                                                                                                                      |
|----------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------|
| `key_agents`         | `str`     | The IDs of the two key vehicles in the interaction: Separated by semicolons (`;`).                                             |
| `path_relation` | `str`     | The driving direction relationship label **before** and **after** the intersection (e.g., `P-M`, `C-O`). <br> - `P-M`: The two agents were running parallel (`P`) before the intersection and merged (`M`) after the intersection. <br> - `C-O`: The two agents were running crossed (`C`) before the intersection and opposite (`O`) after the intersection. |
| `turn_label`         | `str`     | The turning direction of the two vehicles: Recorded in the `td_i-td_j` format, where `td_i` and `td_j` represent the turning directions, each being one of: <br> - `S` (straight) <br> - `L` (left turn) <br> - `R` (right turn) <br> - `U` (U-turn). |
| `priority_label`     | `str`     | The ID of the vehicle with right of priority among the `key_agents`.                                                           |

### Dataset update

`data/3_paperplot_data/all_results.csv` has been updated with AG2 / Argoverse 2 motion forecasting interactions (`av2_motion_forecasting`). During this update, candidate interactions from all datasets were checked with oriented vehicle bounding boxes because AG2 introduced many physical-overlap cases.

The bbox collision check reads vehicle `length` and `width` from the cached agent data when available; otherwise it uses a default passenger-car size of 4.5 m by 1.8 m. Each box is centered on the recorded vehicle position and aligned exactly with the cached heading by default. If the two key-agent boxes have positive overlap area in any checked frame, the row is marked as a collision and excluded from the updated `all_results.csv`.

For the full dataset information, please refer to [Dataset Information](dataset.md#dataset-information).

## To Do
- [ ] Supplementary material, video, slides
- [ ] Update `all_results.csv` with traceability for other datasets
- [x] Update `all_results.csv` with AG2 / Argoverse 2 data and bbox collision filtering 20260519
- [x] Update `all_results.csv` with nuplan traceability to original dataset 20251215
- [x] Our paper published in _Scientific Data_ 20250701
- [x] Supplementary code of the interactive label 20250410
- [x] Preprint paper release 20241128
- [x] Visualization scripts 20241106
- [x] Installation tutorial 20241106
- [x] Initial repo 20241019

## Quick Start

### Environment Setup

Ensure the following prerequisites are satisfied. We recommend using conda for Python environment management.

* Create and activate a conda environment:
  ```bash
  conda create --name interhub python=3.8
  conda activate interhub
  ```

* Upgrade pip to the latest version:
  ```bash
  python -m pip install --upgrade pip
  ```

* (Optional) Change pip source for faster installation if encountering network issues:
  ```bash
  pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  ```

* Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Walk through InterHub with a mini dataset

* Install required trajdata package for the mini dataset from INTERACTION:

  ```bash
  pip install "trajdata[interaction]"
  ```

* A subset of the original *interaction_multi* dataset is provided in `data/0_origin_datasets/interaction_multi` for a quick try. Unify the data with:
  ```bash
  python 0_data_unify.py --desired_data interaction_multi --load_path data/0_origin_datasets/interaction_multi --save_path data/1_unified_cache
  ```

* Extract interaction events from the subset:
  ```bash
  python 1_interaction_extract.py --desired_data interaction_multi --cache_location data/1_unified_cache --save_path data/2_extracted_results
  ```

* Visualize the interaction events:
  ```bash
  python 2_case_visualize.py --cache_location data/1_unified_cache/interaction_multi --interaction_idx_info data/2_extracted_results/results.csv --top_n 3
  ```
  See `figs/case` for visualization results.

## Full Working Flows with InterHub

### 1. Data Unification

**1.1 For ready-to-use interaction data in InterHub**, download and use the data from [InterHub](https://figshare.com/articles/dataset/_b_InterHub_A_Naturalistic_Trajectory_Dataset_with_Dense_Interaction_for_Autonomous_Driving_b_/27899754), unzip it, and place it in the `data/1_unified_cache` folder. Proceed to [2. Interaction Event Extract](#2-interaction-event-extract).

**1.2 For working from scratch or extracting from other data resources**, the origin datasets are necessary. Refer to [dataset.md](dataset.md) for details on building the needed dataset structure. Preprocess the dataset to form a data cache if using initial or unprocessed datasets.

For datasets including **INTERACTION, nuPlan, Waymo, lyft**, `0_data_unify.py` provides scripts for preprocessing raw data into a unified data cache. The project [trajdata](https://github.com/NVlabs/trajdata?tab=readme-ov-file#data-preprocessing-optional) is used in this step. Replace arguments according to the dataset you want to process:

- **desired_data**: List of datasets to process, e.g., `["interaction_multi"]`. See support list in [dataset.md](dataset.md).

- **load_path**: Path where raw data is stored, e.g., `'data/0_origin_datasets/interaction_multi'`.

- **cache_location**: Path where the generated cache will be stored. Ensure enough memory, e.g., `'data/1_unified_cache/interaction_multi'`.

```bash
python 0_data_unify.py \
--desired_data 'waymo_train' \
--load_path path/to/your/dataset \
--save_path path/to/your/dataset/cache \
--use_multiprocessing \
--processes 14
```

### 2. Interaction Event Extraction

```bash
python 1_interaction_extract.py \
--desired_data dataset_name \
--cache_location path/to/your/dataset/cache \
--save_path path/to/save/your/result \
--timerange=5
```

Replace `cache_location` and `save_path` with your paths. By default, a subset of `interaction_multi` dataset is read from the `data/1_unified_cache` folder.

---

### 3. Visualization

#### Case visualization
Run `2_case_visualize.py` to plot interaction segments and generate GIFs.

#### Paper plot
Run `3_paper_plot.py` to plot results in the paper using metadata of interaction events in the full InterHub dataset.

#### Bbox collision check
Run the optional bbox collision checker against an interaction metadata CSV and the corresponding trajdata cache:

```bash
python scripts/check_index_bbox_collisions.py \
--input-csv data/3_paperplot_data/all_results.csv \
--cache-root data/1_unified_cache \
--output-dir data/3_paperplot_data/bbox_collision_check
```

The checker writes the full checked CSV, a no-collision CSV, a collision CSV, an error CSV, and a JSON summary.

---


## Acknowledgment & Disclaimer

This project, InterHub, incorporates code from [trajdata](https://github.com/NVlabs/trajdata), developed by NVIDIA Research. We are not affiliated with NVIDIA or the contributors of trajdata. We extend our sincere gratitude to the trajdata team for their outstanding work in simplifying trajectory data processing.

The use of trajdata code in this project is in accordance with their original license terms. All rights and credits for the trajdata components belong to their respective owners.

## Citation
If you find this repository useful for your research, please consider giving us a star 🌟 and citing our paper.

```bibtex

@article{jiang_naturalistic_2025,
	title = {A naturalistic trajectory dataset with dense interaction for autonomous driving},
	volume = {12},
	issn = {2052-4463},
	url = {https://doi.org/10.1038/s41597-025-05344-7},
	doi = {10.1038/s41597-025-05344-7},
	journal = {Scientific Data},
	author = {Jiang, Xiyan and Zhao, Xiaocong and Liu, Yiru and Li, Zirui and Hang, Peng and Xiong, Lu and Sun, Jian},
	month = jul,
	year = {2025},
	pages = {1084},
}

