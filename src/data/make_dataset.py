import requests
import os
from time import sleep
from progress.bar import Bar
import pandas as pd


# Query Xeno-Canto database and create a metadata file of available recordings
def download_database_metadata(
    country='argentina',
    since='2000-01-01',
    group='birds',
    length='10-60',
    dataset_location='./datasets/xeno_canto_birds/',
):
    api_url = 'https://xeno-canto.org/api/2/recordings?query='
    params = f"cnt:{country}+grp:{group}+len:{length}+since:{since}"

    # Get response from server
    response = requests.get(api_url + params)

    if response.status_code == 200:
        query_data = response.json()
        n_rec = query_data["numRecordings"]
        n_pages = query_data["numPages"]
        print(f"\n• Query result: status-code {response.status_code}. Everything OK!\n")
        print(f"• Found {n_rec} recordings in {n_pages} pages.\n")
    else:
        raise Exception('Server error. Status code {response.status_code}.')

    # Create dataset folder
    try:
        os.makedirs(dataset_location)
        print(f'• Created {dataset_location} folder!\n')
    except:
        print('• Dataset path already exists!\n')

    # Download each page and store them into a Pandas dataframe
    df_list = []
    with Bar('Downloading metadata...', max=n_pages) as bar:
        for page in range(1, n_pages + 1):
            response = requests.get(api_url + params + f"&page={page}")
            data = response.json()
            df = pd.json_normalize(data['recordings'])
            df_list.append(df)
            bar.next()

    # Concatenate all dataframes into one and save to disk
    recordings_dataframe = pd.concat(df_list, ignore_index=True)
    recordings_dataframe.to_csv(os.path.join(dataset_location, 'metadata.csv'), index=False)
    print(f'\n• Saved list of available recordings as {dataset_location}metadata.csv\n')


# Download all audio files from the database that match filter criteria
def filter_metadata(csv_path='./datasets/xeno_canto_birds/metadata.csv', species_count=40, quality=['A']):

    # For mapping length column
    def minutes_to_seconds(mmss_time):
        m, s = mmss_time.split(':')
        return (int(m) * 60) + int(s)

    df = pd.read_csv(
        csv_path + 'metadata.csv',
        usecols=[
            'id',
            'group',
            'gen',
            'sp',
            'ssp',
            'en',
            'loc',
            'type',
            'file',
            'q',
            'length',
            'method',
            'file-name',
        ],
    )

    # Filter metadata and map lenght column to seconds
    df = df.loc[
        (df['group'] == 'birds')
        & (df['gen'] != 'Mystery')
        & (df['sp'] != 'mystery')
        & (df['q'].isin(quality))
        & (df['method'] == 'field recording')
        & (df['type'].isin(['call', 'song']))
    ]
    df['length'] = df['length'].map(minutes_to_seconds)

    # Group df by species, aggregate length, get top species by recording time
    df_group = df.groupby(['gen', 'sp'], as_index=False).sum('length')
    df_group = df_group.sort_values('length', ascending=False).iloc[:species_count]
    top_species = (df_group['gen'] + ' ' + df_group['sp']).values.tolist()

    # Filter recordings based on selected species
    df_filter = df.loc[(df['gen'] + ' ' + df['sp']).isin(top_species)]

    return df_filter


MIN_RECORDED_TIME = 400
SPECIES_COUNT = 40
SAMPLE_RATE = 16000
WINDOW_LENGTH = 3

download_database_metadata()

# df = filter_metadata(species_count=SPECIES_COUNT)
