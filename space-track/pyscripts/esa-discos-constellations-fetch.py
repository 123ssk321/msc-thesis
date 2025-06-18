from pprint import pprint
from time import sleep
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


def fetch_satellite_data(base_url, token, sat_ids, page_size):
    response = requests.get(
        url=f'{base_url}/api/objects',
        headers={
            'Authorization': f'Bearer {token}',
            'DiscosWeb-Api-Version': '2',
        },
        params={
            'filter': f"in(satno,({','.join(sat_ids)}))",
            'include': 'constellations',
            'sort':'satno',
            'page[size]': page_size
        },
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        pprint(response.json())
        return None

def satellite_data(json_data):
    sat_data = []
    for obj in json_data['data']:
        sat_attributes = obj['attributes']
        constellation_ids = [rel_data['id'] for rel_data in obj['relationships']['constellations']['data']]
        sat_attributes['constellationDiscosID'] = ','.join(constellation_ids) if len(constellation_ids) > 0 else None
        sat_data.append(sat_attributes)
    return sat_data

def constellation_data(json_data):
    constell_data = []
    if 'included' in json_data:
        for constellation in json_data['included']:
            constell_attributes = constellation['attributes']
            constell_attributes['discosID'] = constellation['id']
            constell_data.append(constell_attributes)
    return constell_data


def main():
    URL = 'https://discosweb.esoc.esa.int'
    token = '<token>'

    ids_df = pd.read_csv('../datasets/space-track-dataset.csv', usecols=['NORAD_CAT_ID'], memory_map=True)
    ids = list(map(str, ids_df['NORAD_CAT_ID'].unique().tolist()))
    num_satellites = len(ids)
    print(f'Number of satellites:{num_satellites}')

    page_size = 100
    sat_data = []
    constell_data = []
    for i in tqdm(range(0, num_satellites, page_size)):
        json_data = fetch_satellite_data(URL, token, ids[i:i+page_size], page_size)
        if json_data:
            sat_data.extend(satellite_data(json_data))
            constell_data.extend(constellation_data(json_data))
        else:
            exit()
        sleep(5)

    sat_data_df = pd.DataFrame(sat_data)
    print(f'\nSatellite dataframe: {sat_data_df.shape[0]} lines')
    if sat_data_df.shape[0] != num_satellites:
        A = sat_data_df['satno'].unique()
        B = ids_df['NORAD_CAT_ID'].unique()

        not_common_in_A = np.setdiff1d(A, B)
        not_common_in_B = np.setdiff1d(B, A)

        not_common = np.concatenate((not_common_in_A, not_common_in_B))
        print(f"Elements from ESA DISCOS dataset missing in Space-Track dataset:{not_common_in_A}")
        print(f"Elements from Space-Track dataset missing in ESA DISCOS dataset:{not_common_in_B}")
        print(f"Elements not common in both arrays:{not_common}")
    sat_data_df.to_csv('../datasets/esa-discos-satellite-data.csv', index=False)

    constell_data_df = pd.DataFrame(constell_data)
    print(f'\nConstellation dataframe: {constell_data_df.shape[0]} lines')
    constell_data_df.drop_duplicates(ignore_index=True, inplace=True)
    constell_data_df.to_csv('../datasets/esa-discos-constellation-data.csv', index=False)

if __name__ == "__main__":
    main()