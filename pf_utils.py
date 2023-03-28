import math
import pandas as pd

import numpy as np
# import quaternion

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x1 - x2, y1 - y2)

def angle_between(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.atan2(y2 - y1, x2 - x1)

def normalize_angle(a):
    while a > math.pi:
        a -= 2*math.pi

    while a <= -math.pi:
        a += 2*math.pi
    
    return a

def downsample_enus(xs, ys, ts, delta_distance = 0.25, delta_time = 0.2):
    dxs = [xs[0]]
    dys = [ys[0]]
    dts = [ts[0]]

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        t = ts[i]
        if distance((x, y), (dxs[-1], dys[-1])) > delta_distance or abs(t - dts[-1] > delta_time):
            dxs.append(x)
            dys.append(y)
            dts.append(t)

    dxs.append(xs[len(xs)-1])
    dys.append(ys[len(ys)-1])
    dts.append(ts[len(ts)-1])
    
    return dxs, dys, dts

def extract_control(xs, ys, ts):
    # TODO: option to skip some indices?
    angle = angle_between((xs[0], ys[0]), (xs[1], ys[1]))
    # TODO: it is horrible to save the first pose here
    controls = [(xs[0], ys[0], angle, ts[0])]

    for i in range(len(xs) - 1):
        x, y, t = xs[i], ys[i], ts[i]
        next_x, next_y, next_t = xs[i+1], ys[i+1], ts[i+1]
        next_angle = angle_between((x, y), (next_x, next_y))
        angle_diff = normalize_angle(next_angle - angle)
        angle = next_angle
        d = distance((x, y), (next_x, next_y))
        ctrl = d, 0, angle_diff, next_t
        controls.append(ctrl)

    return controls

def read_pf_data(path, truncated):
    if truncated:
        traj = pd.read_csv(path + '/trajectory_truncated.csv')
    else:
        traj = pd.read_csv(path + '/trajectory.csv')
    xs = traj['x_avg[m]']
    ys = traj['y_avg[m]']
    ts = traj['t[s]']

    init_pf = pd.read_csv(path + '/trajectory_init_pf.csv').to_dict('records')

    return {
        'trajectory': downsample_enus(xs, ys, ts),
        'init_pf': init_pf
    }

def read_known_locations(path):
    return pd.read_csv(path + '/trajectory_coords.csv')

def extract_control_and_measurements(path):
    mf = pd.read_csv(path + '/magnetic_field_normalized.csv')
    tr = pd.read_csv(path + '/trajectory_truncated.csv')
    x, y, t = tr['x_avg[m]'], tr['y_avg[m]'], tr['t[s]']
    mx, my, mz = mf['mx'], mf['my'], mf['mz']
    mt = mf['t']
    xd, yd, td = downsample_enus(x, y, t)
    
    for reverse in [False, True]:
        suffix = '' 
        if reverse:
            suffix = '_reversed'
            xd.reverse(), yd.reverse(), td.reverse()

        ctrls = extract_control(xd, yd, td)
        cx, cy, ca, _ = zip(*ctrls)
        ix = np.searchsorted(mt, td)

        if reverse:
            max_t = max(td)
            td = [abs(t - max_t) for t in td]
        
        df = pd.DataFrame()
        df['t'] = td
        df['cx'] = cx
        df['cy'] = cy
        df['ca'] = ca
        df['mx'] = [mx[i] for i in ix]
        df['my'] = [my[i] for i in ix]
        df['mz'] = [mz[i] for i in ix]
        df.to_csv(path + f'/control_and_mf_measurements{suffix}.csv', index=False)


    