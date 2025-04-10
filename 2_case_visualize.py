import os
import pandas as pd
import imageio
from trajdata import UnifiedDataset
from utils.visualize_utils import draw_pic

# define parameters
starting_extension_time = 0.5  # starting extension time in seconds
ending_extension_time = 0       # ending extension time in seconds
considertime = 5  # future time of trajectories in seconds
gif_path = "figs/case"  # path to save gif files
cache_location = 'data/1_unified_cache'   # path to cache location
interaction_idx_info = 'data/2_extracted_results/results.csv'  # path to interaction csv file
top_n = 10  # number of gifs to generate based on the top N intensity values


# read the csv file containing interaction information
extract_df = pd.read_csv(interaction_idx_info)

# get the top N rows with the highest intensity values
top_rows = extract_df.nlargest(top_n, 'intensity')

# iterate through the top N rows
for rank, (idx, row) in enumerate(top_rows.iterrows(), start=1):
    # extract information from the selected row
    desired_data = row['dataset']
    raw_scene_id = int(row['scenario_idx'])
    start = int(row['start'])
    end = int(row['end'])
    interact_ids = row['track_id'].split(';')
    intensity = row['intensity']
    path_relation = row['path_relation']

    # print the processing details
    print(f"Processing Top {rank}: {raw_scene_id}, {start}, {end}, {interact_ids}")

    # initialize the dataset instance and load necessary data
    dataset = UnifiedDataset(
        desired_data=[desired_data],
        standardize_data=False,
        rebuild_cache=False,  # do not rebuild cache
        rebuild_maps=False,   # do not rebuild maps
        centric="scene",
        verbose=True,
        cache_location=cache_location,
        num_workers=os.cpu_count(),
        incl_vector_map=True,
        data_dirs={desired_data: ' '}
    )

    # map scene IDs to their raw data indices
    id_rawid = {desired_scene.raw_data_idx: idx for idx, desired_scene in enumerate(dataset.scenes())}

    # retrieve the desired scene based on the raw scene ID
    desired_scene = dataset.get_scene(id_rawid[raw_scene_id])

    # extract the time step and agent information
    dt = desired_scene.dt
    agents = {agent.name: agent for agent in desired_scene.agents}
    all_agents = list(agents.keys())
    first, last = 99999, 0

    # determine the first and last time steps for the interacting agents
    for agent in interact_ids:
        first = min(first, agents[agent].first_timestep)
        last = max(last, agents[agent].last_timestep)

    print(row, first, last)

    # generate image frames for each timestamp
    pic_list = []
    plot_start = max(first, int(start - starting_extension_time / dt))
    plot_end = min(last, int(end + starting_extension_time / dt))
    all_timesteps = range(plot_start, plot_end)

    # loop through each timestamp to generate images
    for timestamp in all_timesteps:
        draw_pic(
            desired_scene, all_agents, all_timesteps, dataset, id_rawid, raw_scene_id,path_relation,
            interact_ids, timestamp, start, end, dt, considertime, gif_path, 
        )
        
        interact_ids_str = '_'.join(interact_ids)
        pic_list.append(f'{gif_path}/{interact_ids_str}_{timestamp}.png')

    # create gif from the generated images
    gif_file = f"{gif_path}/gif/Top_{rank}_{raw_scene_id}_{start}-{end}_{path_relation}.gif"
    os.makedirs(os.path.dirname(gif_file), exist_ok=True)
    images = [imageio.imread(image_file) for image_file in pic_list]
    imageio.mimsave(gif_file, images, fps=5)

    # optionally remove individual image files after creating the gif
    for image_file in pic_list:
        os.remove(image_file)
