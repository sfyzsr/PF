import yaml
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
from align_utils import *
from pf_utils import *
from particle_filter import *

plt.rcParams["figure.figsize"] = (20, 15)
plt.rcParams["savefig.dpi"] = 300


##################################################################################
# Read yaml config
##################################################################################
with open("gt_matching.yaml", "r") as stream:
    map_info = yaml.safe_load(stream)
trajs = map_info["good_traj"]
pix_to_meter = map_info["cfg"]["pix_to_meter"]
floorplan_scale = map_info["cfg"]["pix_to_meter"]
key_point_locations = map_info["location_dict_pix"]
path_dualfoot_data = map_info["cfg"]["data_path"]
truncate_to_last_timestamp = False
floorplan_image = mpimg.imread(map_info["cfg"]["map_path"])


##################################################################################
# Write keypoint matching information and particle filter init information
##################################################################################
for traj_name in trajs:
    print("handling trajectory:", traj_name)
    # Read data for the trajectory
    traj = map_info[traj_name]
    traj_path = os.path.join(path_dualfoot_data, traj["subdir"])
    key_points = traj["key_points"]
    key_ts = pd.read_csv(os.path.join(traj_path, "saved_timestamps.csv"))
    key_ts["timestamp_global[ms]"] = key_ts["timestamp_global[ms]"] / 1000
    trajectory = pd.read_csv(os.path.join(traj_path, "trajectory.csv"))
    assert len(key_points) == key_ts.shape[0], f"points count doesn't match ts count"

    # Compile matching key points
    match_result = []
    for i in range(len(key_ts)):
        if key_points[i][0] != "z":
            match_result.append(
                [
                    key_point_locations[key_points[i]][0] * pix_to_meter,
                    -key_point_locations[key_points[i]][1] * pix_to_meter,
                    key_ts["timestamp_global[ms]"][i],
                ]
            )

    # Save matched trajectory key points
    match_result = np.array(match_result)
    df = pd.DataFrame()
    df["x"] = match_result[:, 0]
    df["y"] = match_result[:, 1]
    df["t"] = match_result[:, 2]
    df.to_csv(os.path.join(traj_path, "trajectory_coords.csv"))

    # Organize original key points
    traj_keyindex = np.searchsorted(trajectory["t[s]"], key_ts)
    traj_keypoints = pd.DataFrame()
    traj_keypoints["x"] = np.array(
        [trajectory["x_avg[m]"][i] for i in traj_keyindex]
    ).flatten()
    traj_keypoints["y"] = np.array(
        [trajectory["y_avg[m]"][i] for i in traj_keyindex]
    ).flatten()
    traj_keypoints["t"] = np.array(
        [trajectory["t[s]"][i] for i in traj_keyindex]
    ).flatten()

    # Initiate particle filter
    first0 = np.array(traj_keypoints.iloc[0, :2])
    last0 = np.array(traj_keypoints.iloc[1, :2])
    first1 = np.array(match_result[0][:2])
    last1 = np.array(match_result[1][:2])
    v0 = last0 - first0
    v1 = last1 - first1
    d0 = np.linalg.norm(v0)
    d1 = np.linalg.norm(v1)
    a0 = np.arctan2(*v0)
    a1 = np.arctan2(*v1)
    init_pf = {"x": first1[0], "y": first1[1], "scale": d1 / d0, "heading": a0 - a1}
    print(init_pf)
    df_pf = pd.DataFrame(init_pf, index=[0])
    df_pf.to_csv(os.path.join(traj_path, "trajectory_init_pf.csv"), index=False)


##################################################################################
# Particle Filter Align Trajectories
##################################################################################
def location_likelihood(p, known_x, known_y):
    d = math.hypot(p.x - known_x, p.y - known_y)
    dev = 1.0
    return math.exp(-0.5 * (d * d) / (dev * dev)) + 0.0001


def plot_histories(cloud):
    for p in cloud.particles:
        plt.plot(*zip(*p.history), "-")
    # plt.axis('equal')
    plt.show()


clouds = []
trajectories_aligned = []

dev = (0.05, 0.001, 0.02)  # sx, sy, sa
dev_roughen = (0, 0, math.pi * 0.05)
p_count = 2000


for traj_name in trajs:
    traj = map_info[traj_name]
    path = os.path.join(path_dualfoot_data, traj["subdir"])
    print("Processing Trajectory", traj_name, "at", traj["subdir"])
    data = read_pf_data(path, truncate_to_last_timestamp)
    viz_dir = os.path.join(path, "align_viz")
    if not os.path.exists(viz_dir):
        os.mkdir(viz_dir)

    known_locations = read_known_locations(path)
    xs, ys, ts = data["trajectory"]
    init_pf = data["init_pf"][0]
    controls = extract_control(xs, ys, ts)

    cloud = Cloud(p_count, Particle(0, 0, 0))
    clouds.append(cloud)

    fixed = False

    init_move = (init_pf["x"], init_pf["y"], init_pf["heading"])

    cloud.set_pose(
        controls[0][0] + init_move[0],
        controls[0][1] + init_move[1],
        controls[0][2] + init_move[2],
    )
    cloud.record_history()

    for p in cloud.particles:
        p.scale = init_pf["scale"] + random.gauss(0, 0.1)

    location_index = 0

    for ctrl in controls[1:]:
        # print(ctrl)
        t = ctrl[3]
        cloud.move(ctrl, dev)
        cloud.record_history()

        if location_index >= len(known_locations.t):
            location_index = 0
        t_location = known_locations.t[location_index]
        if abs(t - t_location) < 0.25:
            fixed = True
            known_x = known_locations.x[location_index]
            known_y = known_locations.y[location_index]
            location_index += 1
            print("Fixing location", known_x, known_y, t)
            plt.figure()
            plot_image_scale(plt.gca(), floorplan_image, floorplan_scale)
            plot_histories(cloud)
            # plt.show()
            plt.savefig(os.path.join(viz_dir, str(int(t)) + ".png"))
            plt.clf()

            cloud.update_weights(
                partial(location_likelihood, known_x=known_x, known_y=known_y)
            )
            if cloud.neff_ratio() < 0.8:
                print("Resampling")
                cloud.resample()
                cloud.roughen(dev_roughen)
                for p in cloud.particles:
                    p.scale += random.gauss(0, 0.1)

    plt.figure()
    plot_image_scale(plt.gca(), floorplan_image, floorplan_scale)
    plt.plot(known_locations.x, known_locations.y, "or", markersize=10)
    plt.plot(*zip(*cloud.mean_history()), ".", markersize=3)
    # plt.axis('equal')
    plt.savefig(os.path.join(viz_dir, "final.png"))
    plt.clf()

    print("Saving results")
    ts_to_save = [c[3] for c in controls]
    xs_to_save = [c[0] for c in cloud.mean_history()]
    ys_to_save = [c[1] for c in cloud.mean_history()]
    data_to_save = {"x": xs_to_save, "y": ys_to_save, "t": ts_to_save}
    trajectories_aligned.append(data_to_save)
    df = pd.DataFrame(data_to_save)
    df.to_csv(path + "/trajectory_aligned.csv", index=False)

