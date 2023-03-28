import os
import json
import math
from re import sub

import pymap3d as pm
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import numpy as np

from functools import partial

from statistics import median


def process_dualfoot_data(path_dualfoot_data, path_dualfoot_lib):
    # next(os.walk(path_dualfoot_data))[1]
    subdirs = os.listdir(path_dualfoot_data)
    for subdir in subdirs:
        if "Pro 1" in subdir and "2022-07-08" in subdir:
            path = path_dualfoot_data + subdir
            aw70_files = [x for x in os.listdir(path) if x[:4] == "AW70"]
            if len(aw70_files) <= 2:
                print("Only one foot sensor data present. Making dummy copies.")
                for file in aw70_files:
                    dummy_name = file.replace("_imu", "dummy_imu")
                    cmd = f'cp "{path}/{file}" "{path}/{dummy_name}"'
                    print(cmd)
                    os.system(cmd)

            print("Processing DualFoot: ", path)
            flags = "--save"
            print("flags:", flags)
            cmd = f'python3 {path_dualfoot_lib}main.py --path="{path}" {flags}'
            os.system(cmd)


def load_floorplan_pgw(path_floor_plan, floor):
    floorplan_pgw = []
    with open(f"{path_floor_plan}/{floor}.pgw") as f:
        for x in f.read().split("\n"):
            floorplan_pgw.append(float(x))
    return floorplan_pgw


# note: currently handles only straight images


def image_enu_data(img, pgw):
    y_pix, x_pix, _ = img.shape
    wgs0 = (pgw[5], pgw[4])
    enu0 = pm.geodetic2enu(*wgs0, 0, *wgs0, 0)
    x_scale = pgw[0]
    y_scale = pgw[3]
    wgs1 = (pgw[5] + y_pix * y_scale, pgw[4] + x_pix * x_scale)[:2]
    enu1 = pm.geodetic2enu(*wgs1, 0, *wgs0, 0)[:2]
    return {
        "enu0": enu0,
        "enu1": enu1,
        "center_x": (enu0[0] + enu1[0]) / 2,
        "center_y": (enu0[1] + enu1[1]) / 2,
        "width": enu1[0] - enu0[0],
        "height": enu0[0] - enu1[0],
    }


def truncate_trajectory_to_last_timestamp(path):
    trajectory = pd.read_csv(path + "/trajectory.csv")
    ts = trajectory["t[s]"].tolist()

    timestamps = pd.read_csv(path + "/saved_timestamps.csv")
    timestamps_sec = [t / 1000 for t in timestamps["timestamp_global[ms]"]]

    max_timestamp = max(timestamps_sec)

    # add one extra to avoid index oob later
    truncation_index = np.searchsorted(ts, max_timestamp) + 1

    truncated_trajectory = trajectory.head(truncation_index)

    truncated_trajectory.to_csv(path + "/trajectory_truncated.csv", index=False)


# note: currently handles only straight images
def plot_image_enu(ax, img, pgw):
    data = image_enu_data(img, pgw)
    enu0 = data["enu0"]
    enu1 = data["enu1"]
    transform = mtransforms.Affine2D().scale(
        enu1[0] / img.shape[1], enu1[1] / img.shape[0]
    )

    im = ax.imshow(img, interpolation="none", origin="lower")
    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    ax.plot([enu0[0], enu1[0]], [enu0[1], enu1[1]], "o")

    ax.axis("equal")
    xlim = (enu0[0], 2 * enu1[0])
    ylim = (enu1[1], enu0[1])
    ax.set(xlim=xlim, ylim=ylim)


def plot_image_scale(ax, img, scale):
    enu0 = [0, 0]
    enu1 = [img.shape[1] * scale, -img.shape[0] * scale]
    transform = mtransforms.Affine2D().scale(scale, -scale)

    im = ax.imshow(img, interpolation="none", origin="lower")
    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    ax.plot([enu0[0], enu1[0]], [enu0[1], enu1[1]], "o")

    ax.axis("equal")
    xlim = (enu0[0], enu1[0])
    ylim = (enu1[1], enu0[1])
    ax.set(xlim=xlim, ylim=ylim)


def prepare_data(trajectory: pd.DataFrame):
    xs = trajectory["x_avg[m]"].tolist()
    ys = trajectory["y_avg[m]"].tolist()
    ts = trajectory["t[s]"].tolist()

    return list(zip(xs, ys, ts))


def find_closest(x, y, xyis):
    min_d = 10000000000
    min_xyi = None

    for xyi in xyis:
        x0, y0, i0 = xyi
        dx = x0 - x
        dy = y0 - y
        d = dx * dx + dy * dy
        if d < min_d:
            min_d = d
            min_xyi = xyi

    return min_xyi


def save_collected_data(state):
    output_path = state["output_path"]
    xyts = state["new_xyis"]
    xs, ys, ts = map(list, zip(*xyts))
    df = pd.DataFrame()
    df = df.assign(x=xs)
    df = df.assign(y=ys)
    df = df.assign(t=ts)
    df.to_csv(output_path + "/trajectory_coords.csv")

    df_pf = pd.DataFrame(state["init_pf_state"], index=[0])
    df_pf.to_csv(output_path + "/trajectory_init_pf.csv", index=False)


def plot_image_and_trajectory(state):
    state["fig"] = plt.gcf()
    state["ax"] = plt.gca()
    state["fig"].clf()
    state["ax"] = state["fig"].add_subplot(111)
    plot_image_enu(state["ax"], state["floorplan_image"], state["floorplan_pgw"])

    # enlarge the axis to enclose the trajectory bb
    xs = [x[0] for x in state["xyis"]]
    ys = [x[1] for x in state["xyis"]]
    x1, x2, y1, y2 = plt.axis()
    plt.axis((min(x1, min(xs)), max(x2, max(xs)), min(y1, min(ys)), max(y2, max(ys)),))

    state["ax"].plot(xs, ys, ".", markersize=0.5)
    state["ax"].plot(
        [state["xyis"][i][0] for i in state["ts_ixs_tr"]],
        [state["xyis"][i][1] for i in state["ts_ixs_tr"]],
        "o",
        markersize=4,
    )
    plt.title(state["title_text"])


def onclick(event, state):
    plot_image_and_trajectory(state)
    advance = False

    def handle_trajectory_input():
        state["close_first"] = None
        state["close_last"] = None
        state["title_text"] = "Selected trajectory point"
        state["closest"] = find_closest(event.xdata, event.ydata, state["xyis"])

        first_x, first_y = state["xyis"][0][0:2]
        if (
            math.hypot(state["closest"][0] - first_x, state["closest"][1] - first_y)
            < 1.0
        ):
            state["title_text"] += " (close to first)"
            state["close_first"] = state["xyis"][0]

        last_x, last_y = state["xyis"][-1][0:2]
        if math.hypot(state["closest"][0] - last_x, state["closest"][1] - last_y) < 1.0:
            state["title_text"] += " (close to last)"
            state["close_last"] = state["xyis"][-1]

        state["ax"].plot(state["closest"][0], state["closest"][1], "o", markersize=7)

    def handle_ground_truth_input():
        def handle_closest(closest, st=""):
            state["ax"].plot(closest[0], closest[1], "o", markersize=7)
            state["ax"].plot(event.xdata, event.ydata, "o", markersize=7)
            state["orig_xyis"].append(closest)
            state["new_xyis"].append((event.xdata, event.ydata, closest[2]))
            state["title_text"] = (
                "Selected GT point " + st + " " + str(len(state["new_xyis"]))
            )

        if not (state["close_first"] or state["close_last"]):
            handle_closest(state["closest"])
        else:
            if state["close_first"]:
                handle_closest(state["close_first"])

            if state["close_last"]:
                handle_closest(state["close_last"])

    if event.button == 1:
        handle_trajectory_input()

    if event.button == 2:
        if state["new_xyis"]:
            state["init_pf_state"] = initial_pf_state(
                state["orig_xyis"], state["new_xyis"]
            )
            save_collected_data(state)
            state["title_text"] = "Saved"
            state["title_text"] += " " + str(len(state["new_xyis"]))
        else:
            state["title_text"] = "Skipping"

        advance = True

    if event.button == 3 and state["closest"]:
        handle_ground_truth_input()

    for i in range(len(state["orig_xyis"])):
        state["ax"].plot(
            [state["orig_xyis"][i][0], state["new_xyis"][i][0]],
            [state["orig_xyis"][i][1], state["new_xyis"][i][1]],
            "-",
        )

    plt.title(state["title_text"])
    if advance:
        collect_trajectory_data(state)


def collect_trajectory_data(state):
    subdirs = os.listdir(state["path_dualfoot_data"])
    if state["trajectory_index"] >= len(subdirs):
        state["fig"].canvas.mpl_disconnect(state["cid"])
        return
    state["closest"] = None
    subdir = subdirs[state["trajectory_index"]]
    path = state["path_dualfoot_data"] + "/" + subdir

    state["title_text"] += " " + subdir

    state["orig_xyis"] = []
    state["new_xyis"] = []
    state["output_path"] = path

    trajectory_filename = ""
    if state["truncated"]:
        trajectory_filename = path + "/trajectory_truncated.csv"
    else:
        path + "/trajectory.csv"

    trajectory = pd.read_csv(trajectory_filename)

    timestamps = pd.read_csv(path + "/saved_timestamps.csv")
    xyis = prepare_data(trajectory)

    mdx = median([x[0] for x in xyis])
    mdy = median([x[1] for x in xyis])

    image_data = image_enu_data(state["floorplan_image"], state["floorplan_pgw"])
    cx = image_data["center_x"]
    cy = image_data["center_y"]

    state["xyis"] = list(
        zip(
            [x[0] + (cx - mdx) + image_data["width"] for x in xyis],
            [x[1] + (cy - mdy) for x in xyis],
            [x[2] for x in xyis],
        )
    )

    ts = [t / 1000 for t in timestamps["timestamp_global[ms]"]]

    state["ts_ixs_tr"] = np.searchsorted(trajectory["t[s]"], ts)

    plt.ion()

    if not state["fig"]:
        state["fig"], state["ax"] = plt.subplots(1, 1)
        state["cid"] = state["fig"].canvas.mpl_connect(
            "button_press_event", partial(onclick, state=state)
        )

    plot_image_and_trajectory(state)
    # plt.axis('equal')
    plt.show()

    state["trajectory_index"] += 1


def init_collection_state(
    path_dualfoot_data, floorplan_image, floorplan_pgw, truncate_to_last_timestamp
):
    return {
        "path_dualfoot_data": path_dualfoot_data,
        "floorplan_image": floorplan_image,
        "floorplan_pgw": floorplan_pgw,
        "truncated": truncate_to_last_timestamp,
        "trajectory_index": 0,
        "xyis": [],
        "tr_ixs_tr": [],
        "orig_xyis": [],
        "new_xyis": [],
        "closest": None,
        "close_first": None,
        "close_last": None,
        "fig": None,
        "ax": None,
        "cid": None,
        "title_text": "Align data",
    }


def initial_pf_state(orig_xyts, new_xyts):
    # TODO: rething what index to use
    # TODO: these must be sorted!
    orig_xyts.sort(key=lambda x: x[2])
    new_xyts.sort(key=lambda x: x[2])

    first0 = np.array(orig_xyts[0][:2])
    last0 = np.array(orig_xyts[1][:2])
    first1 = np.array(new_xyts[0][:2])
    last1 = np.array(new_xyts[1][:2])

    v0 = last0 - first0
    v1 = last1 - first1

    d0 = np.linalg.norm(v0)
    d1 = np.linalg.norm(v1)

    a0 = np.arctan2(*v0)
    a1 = np.arctan2(*v1)

    return {"x": first1[0], "y": first1[1], "scale": d1 / d0, "heading": a0 - a1}


def trajectory_coordinates_to_fusion_case(
    path, accuracy, step_lenght_error, orientation_error
):
    output_path = path + "/fusion_case"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_csv(path + "/trajectory_coords.csv")
    df.sort_values("t", inplace=True)
    xs = df.x.tolist()
    ys = df.y.tolist()
    ts = df.t.tolist()

    ts_ms = [int(t * 1000) for t in ts]
    location_9 = pd.DataFrame()
    location_9 = location_9.assign(time=ts_ms)
    location_9 = location_9.assign(x=xs)
    location_9 = location_9.assign(y=ys)
    location_9["accuracy (m)"] = accuracy
    location_9.to_csv(output_path + "/location_9.csv", index=False)

    df = pd.read_csv(path + "/trajectory.csv")
    ts = df["t[s]"].tolist()
    xs = df["x_avg[m]"].tolist()
    ys = df["y_avg[m]"].tolist()

    ts_ms = ts_ms = [int(t * 1000) for t in ts]
    pdr_nn_sensors = pd.DataFrame()
    pdr_nn_sensors["timestamp[ms]"] = ts_ms
    pdr_nn_sensors["x"] = xs
    pdr_nn_sensors["y"] = ys
    pdr_nn_sensors["step_length_error"] = step_lenght_error
    pdr_nn_sensors["orientation_error"] = orientation_error

    pdr_nn_sensors.to_csv(output_path + "/pdr_nn_sensors.csv", index=False)

    empty_traj = pd.DataFrame()
    empty_traj["tmsp"] = [0]
    empty_traj["x_avg"] = [0]
    empty_traj["y_avg"] = [0]

    empty_traj.to_csv(output_path + "/trajectory.csv")


def trajectory_coordinates_to_fusion_case_all(
    path_dualfoot_data, accuracy, step_lenght_error, orientation_error
):
    for subdir in os.listdir(path_dualfoot_data):
        print("Parsing data for fusion:", path_dualfoot_data + subdir)
        trajectory_coordinates_to_fusion_case(
            path_dualfoot_data + subdir, accuracy, step_lenght_error, orientation_error
        )


def process_fusion_data(path, path_fusion_binary):
    path_fusion_data = path + "/fusion_case"
    flags_init = (
        "--std-initial-x 1000.0 --std-initial-y 1000.0 --std-initial-orientation 100.0"
    )
    # flags = '--global-parameters-only -w 3000 --disable-ransac --tails-only -oc'
    flags = "-w 3000 --disable-ransac --tails-only -oc"
    output_file = f"{path_fusion_data}/output_fusion.json"
    cmd = f'java -jar {path_fusion_binary} -d "{path_fusion_data}" -o "{output_file}" {flags} {flags_init}'
    os.system(cmd)


def process_fusion_data_all(path_dualfoot_data, path_fusion_binary):
    for subdir in os.listdir(path_dualfoot_data):
        print("Processing fusion data:", path_dualfoot_data + subdir)
        process_fusion_data(path_dualfoot_data + subdir, path_fusion_binary)


def visualize_output(path, floorplan_image, floorplan_pgw):
    path_fusion_data = path + "/fusion_case"
    output_file = f"{path_fusion_data}/output_fusion.json"

    corr_data = []
    with open(output_file) as f:
        corr_data = json.load(f)

    pos_windows = corr_data["position-windows"]
    window = pos_windows[-1]

    xs = [p["x"] for p in window]
    ys = [p["y"] for p in window]

    print("Vals from windows", len(xs))
    print("Vals from corr pos", len(corr_data["corrected-pdr-positions"]))

    measurements = corr_data["position-measurement-data"]
    xs_meas = [p["x"] for p in measurements]
    ys_meas = [p["y"] for p in measurements]

    plt.figure()
    plot_image_enu(plt.gca(), floorplan_image, floorplan_pgw)
    plt.plot(xs_meas, ys_meas, ".r", markersize=30)
    plt.plot(xs, ys, ".")
    plt.show()

    return (xs, ys)


def visualize_output_all(path_dualfoot_data, floorplan_image, floorplan_pgw):
    xs_all = []
    ys_all = []
    for subdir in os.listdir(path_dualfoot_data):
        print("Visualizing output from:", path_dualfoot_data + subdir)
        xs, ys = visualize_output(
            path_dualfoot_data + subdir, floorplan_image, floorplan_pgw
        )
        xs_all.extend(xs)
        ys_all.extend(ys)

    plt.figure()
    plot_image_enu(plt.gca(), floorplan_image, floorplan_pgw)

    plt.plot(xs_all, ys_all, ".")


def normalize_magnetic(path):
    from scipy.spatial.transform import Rotation as R

    rv = pd.read_csv(path + "/rotation_vector_16.csv", skiprows=1)
    qx, qy, qz, qa, qt = (
        rv["x*sin(a/2)[]"],
        rv["y*sin(a/2)[]"],
        rv["z*sin(a/2)[]"],
        rv["cos(a/2)[]"],
        rv["imuTimestamp"],
    )

    mf = pd.read_csv(path + "/magnetic_field_5.csv", skiprows=1)
    mx, my, mz, mt = (
        mf["mfield_x[uT]"],
        mf["mfield_y[uT]"],
        mf["mfield_z[uT]"],
        mf["imuTimestamp"],
    )

    corresponding_indices = np.searchsorted(mt, qt)

    mfn = []
    for i in range(len(mx)):
        x, y, z, t = mx[i], my[i], mz[i], mt[i]
        q_ix = corresponding_indices[i]
        q = [qx[q_ix], qy[q_ix], qz[q_ix], qa[q_ix]]
        r = R.from_quat(q)
        m = [x, y, z]
        mfn.append(r.apply(m))

    mxn, myn, mzn = zip(*mfn)

    normalized_data = pd.DataFrame()
    t_sec = [(t - mt[0]) / 1000.0 for t in mt]
    normalized_data["t"] = t_sec
    normalized_data["mx"] = mxn
    normalized_data["my"] = myn
    normalized_data["mz"] = mzn

    normalized_data.to_csv(path + "/magnetic_field_normalized.csv", index=False)

    plt.figure()
    plt.plot(mx, label="x")
    plt.plot(my, label="y")
    plt.plot(mz, label="z")
    plt.plot(mxn, ":", label="xn")
    plt.plot(myn, ":", label="yn")
    plt.plot(mzn, ":", label="zn")
    plt.title(path)
    plt.legend()
    plt.show()

    plt.figure()
    mxy = [math.sqrt(x * x + y * y) for x, y in zip(mx, my)]
    plt.plot(mxy, label="xy")
    plt.plot(mz, label="z")
    mxyn = [math.sqrt(x * x + y * y) for x, y in zip(mxn, myn)]
    plt.plot(mxyn, ":", label="xyn")
    plt.plot(mzn, ":", label="zn")
    plt.title(path)
    plt.legend()
    plt.show()


def create_magnetic_field_map(path_dualfoot_data):
    subdirs = os.listdir(path_dualfoot_data)
    all_data = []
    for subdir in subdirs:
        path = path_dualfoot_data + subdir
        trajectory = pd.read_csv(path + "/trajectory_aligned.csv")
        mf = pd.read_csv(path + "/magnetic_field_normalized.csv")
        # Running through rolling average filter
        mxf = mf.mx.rolling(window=5, min_periods=1).mean()
        myf = mf.my.rolling(window=5, min_periods=1).mean()
        mzf = mf.mz.rolling(window=5, min_periods=1).mean()
        mf_xyz = list(zip(mxf, myf, mzf))

        corresponding_indices = np.searchsorted(mf.t, trajectory.t)
        corresponding_mf = [mf_xyz[i] for i in corresponding_indices]
        mx, my, mz = zip(*corresponding_mf)

        data = zip(trajectory.x, trajectory.y, mx, my, mz)
        all_data.extend(data)

    x, y, mx, my, mz = zip(*all_data)
    mxy = [math.sqrt(x * x + y * y) for x, y in zip(mx, my)]
    for m in [mxy, mz]:
        plt.figure()
        clamped_m = [min(x, 75) for x in m]
        plt.scatter(x, y, s=40, c=clamped_m, cmap="viridis")
        plt.axis("equal")
        plt.show()

    df = pd.DataFrame()
    df["x"] = x
    df["y"] = y
    df["mx"] = mx
    df["my"] = my
    df["mz"] = mz

    df.to_csv(path_dualfoot_data + "/../magnetic_map.csv", index=False)

