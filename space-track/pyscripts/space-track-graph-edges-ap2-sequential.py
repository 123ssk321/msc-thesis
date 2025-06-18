import math
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import concurrent.futures

from tqdm import tqdm

import orekit
vm = orekit.initVM()
print ('Java version:',vm.java_version)
print ('Orekit version:', orekit.VERSION)

from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir
cluster = True
orekit_data_path = '/data/f.caldas/ssk/' if cluster else '../../'
setup_orekit_curdir(orekit_data_path+'datasets/orekit-data.zip')
from java.util import Arrays
from orekit import JArray_double

from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.frames import FramesFactory, LOFType, LocalOrbitalFrame
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants


def generate_timestamps(start_date, end_date, step_time_sec):
    duration = (end_date-start_date).total_seconds()  # in seconds

    timestamps = [start_date + timedelta(seconds=dt) for dt in np.arange(0, duration, step_time_sec)]
    return timestamps

def generate_weights(size):
    weights = np.random.rand(size)
    weights[size - 1] += 0.5
    weights.sort()
    weights /= weights.sum()

    return weights

def load_dataset(reduced_frac, reduced_alt_e, reduced_leos, nrows=None, **kwargs):
    reduced, frac1 = reduced_frac
    reduced_sample_alt_e, frac2, min_alt, max_alt, e_thres, sampled1 = reduced_alt_e
    reduced_sample_leos, frac3, leo, sampled2, = reduced_leos
    if reduced:
        filepath = f"../datasets/space-track-dataset-reduced-{int(frac1 * 100)}.csv"
    elif reduced_sample_alt_e:
        if sampled1:
            filepath = f"../datasets/space-track-dataset-reduced-{int(frac2 * 100)}-h-{min_alt}-{max_alt}-e-{int(e_thres * 100)}.csv"
        else:
            filepath = f"../datasets/space-track-dataset-reduced-h-{min_alt}-{max_alt}-e-{int(e_thres * 100)}.csv"
    elif reduced_sample_leos:
        if sampled2:
            filepath = f"../datasets/space-track-dataset-{leo}-reduced-{int(frac3 * 100)}.csv"
        else:
            filepath = f"../datasets/space-track-dataset-{leo}.csv"
    else:
        filepath = '../datasets/space-track-dataset.csv'
    filepath = kwargs.get('data_folder_location', '../') + filepath
    st_df = pd.read_csv(filepath, nrows=nrows, memory_map=True)
    st_df['EPOCH_TIME'] = st_df['EPOCH_TIME'].apply(lambda t: t.split('.')[0])

    esa_filepath = kwargs.get('data_folder_location', '../') + 'datasets/esa-discos-satellite-data.csv'
    esa_df = pd.read_csv(esa_filepath, usecols=['satno', 'constellationDiscosID'],
                         memory_map=True)

    df = pd.merge(st_df, esa_df, how='left', left_on='NORAD_CAT_ID', right_on='satno').drop('satno', axis=1).rename(
        columns={'constellationDiscosID': 'CONSTELLATION_DISCOS_ID'})

    return df

def date_time(epoch_date, epoch_time):
    d = epoch_date.split('-')
    year, month, day = int(d[0]), int(d[1]), int(d[2])

    t = epoch_time.split(':')
    hour, minute, second = int(t[0]), int(t[1]), int(t[2])
    return datetime(year, month, day, hour, minute, second)

def combine_date_time(row):
    return date_time(row[4], row[5])

def duration_from(date, target):
    return (date-target).total_seconds()

def absolute_date_from(date_time):
    return AbsoluteDate(date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute, float(date_time.second), TimeScalesFactory.getUTC())

def rev_day_from(rad_per_sec):
    revolutions_per_day = (rad_per_sec * 24 * 3600) / (2 * math.pi)
    return revolutions_per_day

def rev_day2_from(rad_per_sec2):
    revolutions_per_day2 = (rad_per_sec2 * (24 * 3600) ** 2) / ((2 * math.pi) ** 2 * 3600)
    return revolutions_per_day2

def degrees_from(radians):
    degrees = radians * (180 / math.pi)
    return degrees

def m_from(km):
    return km * 1000

def km_from(m):
    return m / 1000

def min_from(seconds):
    return seconds / 60


def propagate_tle(tle_line1, tle_line2, date):
    tle = TLE(tle_line1, tle_line2)
    propagator = TLEPropagator.selectExtrapolator(tle)

    state = propagator.propagate(absolute_date_from(date))

    initial_frame = state.getFrame()
    lof = LocalOrbitalFrame(initial_frame, LOFType.QSW, propagator, str(tle.getSatelliteNumber()) + "_lof")
    transformer = initial_frame.getTransformTo(lof, tle.getDate())

    propagated_tle = TLE.stateToTLE(state, tle,
                                    propagator.getDefaultTleGenerationAlgorithm(tle.getUtc(), state.getFrame()))

    return propagated_tle, state, transformer


def update_nodes_data(nodes_data, data):
    for k in nodes_data:
        nodes_data[k].append(data[k])


def timestamp_data(tle_df, ids, timestamp, nodes_data, nodes_data_keys, earth_rad_km):
    t_data = []
    for sat_id in ids:
        sat_df = tle_df[tle_df['NORAD_CAT_ID'] == sat_id]

        sat_ser = sat_df.iloc[0]
        cat_cols = ('NORAD_CAT_ID', 'OBJECT_NAME', 'OBJECT_ID', 'DECAY_DATE', 'CENTER_NAME', 'REF_FRAME', 'TIME_SYSTEM',
                    'MEAN_ELEMENT_THEORY', 'EPHEMERIS_TYPE', 'CLASSIFICATION_TYPE', 'OBJECT_TYPE', 'RCS_SIZE',
                    'CONSTELLATION_DISCOS_ID')
        node_data = {col: sat_ser[col] for col in cat_cols}

        sat_data = (sat_df[['CONSTELLATION_DISCOS_ID', 'NORAD_CAT_ID', 'TLE_LINE1', 'TLE_LINE2', 'EPOCH_DATE',
                            'EPOCH_TIME']]).to_numpy()

        date_time_date_col = np.apply_along_axis(combine_date_time, 1, sat_data)
        sat_data = np.column_stack((sat_data[:, 0:4], date_time_date_col))
        closest_idx = np.argmin(np.abs(np.vectorize(duration_from)(sat_data[:, 4], timestamp)))

        tle, state, transformer = propagate_tle(sat_data[closest_idx, 2], sat_data[closest_idx, 3], timestamp)
        a = km_from(state.getA())  # in km to keep consistent with space-track
        e = tle.getE()
        pv = state.getPVCoordinates()
        pos, vel = pv.getPosition(), pv.getVelocity()

        # orekit units converted to space-track units
        tle_data = [rev_day_from(tle.getMeanMotion()), e, degrees_from(tle.getI()), degrees_from(tle.getRaan()),
                    degrees_from(tle.getPerigeeArgument()), degrees_from(tle.getMeanAnomaly()),
                    tle.getRevolutionNumberAtEpoch(), tle.getBStar(), rev_day2_from(tle.getMeanMotionFirstDerivative()),
                    rev_day2_from(tle.getMeanMotionSecondDerivative()), a, min_from(state.getKeplerianPeriod()),
                    a * (1 + e) - earth_rad_km, a * (1 - e) - earth_rad_km]
        pv_data = [km_from(pos.x), km_from(pos.y), km_from(pos.z), km_from(vel.x), km_from(vel.y), km_from(vel.z),
                   timestamp]
        tle_data.extend(pv_data)

        num_cols = [col for col in nodes_data_keys if col not in cat_cols]
        node_data.update({col: val for col, val in zip(num_cols, tle_data)})

        update_nodes_data(nodes_data, node_data)

        t_data.append((sat_data[0, 0], sat_id, pv, transformer))
    return t_data


def conjunction(pv1, pv2, limits, weighted=False):
    distance = -1
    (r_lim, r_weight), (it_lim, it_weight), (ct_lim, ct_weight) = limits
    pos1 = pv1.getPosition()
    pos2 = pv2.getPosition()
    radial1, in_track1, cross_track1 = pos1.x, pos1.y, pos1.z
    radial2, in_track2, cross_track2 = pos2.x, pos2.y, pos2.z
    r_dist = math.fabs(radial1 - radial2)
    it_dist = math.fabs(in_track1 - in_track2)
    ct_dist = math.fabs(cross_track1 - cross_track2)

    if weighted:
        # WEIGHTED CONDITION
        if r_weight * (r_dist <= r_lim) + it_weight * (it_dist <= it_lim) + ct_weight * (ct_dist <= ct_lim) > 0.5:
            distance = r_weight * (r_dist ** 2) + r_weight * (it_dist ** 2) + r_weight * (ct_dist ** 2)
    else:
        if r_dist <= r_lim or it_dist <= it_lim or ct_dist <= ct_lim:
            distance = (r_dist ** 2) + (it_dist ** 2) + (ct_dist ** 2)

    return r_dist, it_dist, ct_dist, math.sqrt(distance) if distance >= 0 else distance


def update_edges(edges, data):
    for k in edges:
        edges[k].append(data[k])


def check_conjunction(sat1_id, pv1_lof, sat2_id, pv2_lof, limits, date_time, prop, edges):
    updated = False
    r_dist, it_dist, ct_dist, distance = conjunction(pv1_lof, pv2_lof, limits)
    if distance >= 0:
        data = {'source': sat1_id, 'target': sat2_id, 'weight': 1, 'r_dist': r_dist, 'it_dist': it_dist,
                'ct_dist': ct_dist, 'dist': distance, 'datetime': date_time, 'prop': prop}
        update_edges(edges, data)
        updated = True
    return updated


def tle_to_edges_sequential(tle_df, tle_df_cols, timestamps, ids, limits, earth_rad_km):
    edges = {'source': [], 'target': [], 'weight': [], 'r_dist': [], 'it_dist': [], 'ct_dist': [], 'dist': [],
             'datetime': [], 'prop': []}
    nodes_data = {col: [] for col in tle_df_cols if
                  col not in ('EPOCH_DATE', 'EPOCH_TIME', 'CREATION_DATE', 'TLE_LINE1', 'TLE_LINE2')}
    nodes_data.update({col: [] for col in ['PX', 'PY', 'PZ', 'VX', 'VY', 'VZ', 'TIMESTAMP']})

    num_satellites = ids.shape[0]
    for timestamp in tqdm(timestamps):
        t_data = timestamp_data(tle_df, ids, timestamp, nodes_data, list(nodes_data.keys()), earth_rad_km)

        updated = False
        for i in range(num_satellites):
            constellation1_id, sat1_id, pv1, transformer = t_data[i]
            pv1_lof = transformer.transformPVCoordinates(pv1)
            for j in range(i + 1, num_satellites):
                constellation2_id, sat2_id, pv2, _ = t_data[j]
                if constellation1_id == constellation2_id:
                    continue

                pv2_lof = transformer.transformPVCoordinates(pv2)
                updated |= check_conjunction(sat1_id, pv1_lof, sat2_id, pv2_lof, limits, timestamp, True, edges)
        if not updated:
            data = {'source': None, 'target': None, 'weight': None, 'r_dist': None, 'it_dist': None,
                    'ct_dist': None, 'dist': None, 'datetime': timestamp, 'prop': None}
            update_edges(edges, data)

    return pd.DataFrame(edges), pd.DataFrame(nodes_data)
def summary(edges_df):
    datetime_grouped_edges_df = edges_df.groupby('datetime')
    summary_df = datetime_grouped_edges_df.agg(
        num_edges=('source', 'size'),
        num_unique_src=('source', 'nunique'),
        num_unique_tgt=('target', 'nunique')
    )
    summary_df['num_nodes'] = datetime_grouped_edges_df.apply(lambda x: len(set(x['source']) | set(x['target'])),
                                                           include_groups=False)
    summary_df['graph_density'] = datetime_grouped_edges_df.apply(
        lambda x: len(x) / (len(set(x['source']) | set(x['target'])) * (len(set(x['source']) | set(x['target'])) - 1)),
        include_groups=False)
    return summary_df

def main():

    start_date = datetime(2023, 12, 28, 0, 0, 0)
    end_date = datetime(2023, 12, 29, 0, 0, 0)
    step_time_sec = 60 * 60  # hour by hour in seconds

    timestamps = generate_timestamps(start_date, end_date, step_time_sec)
    print(f'Number of timestamps: {len(timestamps)}')

    print('#' * 144)
    print('Loading dataset...')
    reduced_frac = (True, 0.25)
    reduced_alt_e = (False, 1.0, 500, 600, 0.2, False)
    reduced_leos = (False, 1.0, 'leo4', False)  # smallest LEO

    nrows=101
    data_folder_location = orekit_data_path if cluster else '../'
    df = load_dataset(reduced_frac, reduced_alt_e, reduced_leos, nrows=nrows, data_folder_location=data_folder_location)
    print(f'Number of lines: {df.shape[0]}')
    print(df.head())

    print('#' * 144)
    print('Building satellite graph...')
    leo1_limits = (0.4, 44, 51) # in km
    leo2_limits = (0.4, 25, 25)
    leo3_limits = (0.4, 12, 12)
    leo4_limits = (0.4, 2, 2)

    w = generate_weights(3)
    # radial least covariance == more certainty == heavier weight
    # in-track more covariance == least certainty == lighter weight
    limit_weights = (w[2], w[0], w[1])
    print(f'Limit weights: {limit_weights}')

    limits = tuple(map(lambda x: x*1000,  leo4_limits))
    sat_ids = df['NORAD_CAT_ID'].unique()
    sat_ids.sort()

    start_time = time.time()

    num_cores = 4
    edges_df, node_feats_df = tle_to_edges_sequential(df, list(df.columns), timestamps, sat_ids, tuple(zip(limits, limit_weights)),
                                           km_from(Constants.WGS84_EARTH_EQUATORIAL_RADIUS))

    elapsed_time = time.time() - start_time
    print(f'Graph building time: {elapsed_time:.2f} seconds')

    print(f'\nTimestamps without edges:{(edges_df[edges_df["source"].isnull()]["datetime"]).unique()}')
    print(f'Number of total nodes:{sat_ids.shape[0]}')
    print(f'Number of lines in node features dataframe:{node_feats_df.shape[0]}')

    print('#' * 144)
    print('Summary')
    summary_df = summary(edges_df)
    print(summary_df.head())

    print('#' * 144)
    print('Summary Stats')
    print(summary_df.describe())

    print('#' * 144)
    print('END')

if __name__ == "__main__":
    main()