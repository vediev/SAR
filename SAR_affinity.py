import pandas as pd
import numpy as np
import datetime


def affinity(df_usage=None, REF_DATE="max", DECAY=30, p=99, weights=False):
    
    df_usage['Date'] = pd.to_datetime(df_usage['Date'])
    if(REF_DATE == "max"):
        REF_DATE = df_usage['Date'].max().strftime("%Y/%m/%dT%H:%M:%S")
    else:
        NOW = datetime.datetime.now().strftime("%Y/%m/%dT%H:%M:%S")
        REF_DATE = NOW

    print("DEBUG: REF_DATE:",REF_DATE)
    print("DEBUG: DECAY in days:",DECAY)
    if(weights):
        print("DEBUG: Mixed events with different weights each are used! Assumes Input format is: <UserId, ItemId, TimeStamp, Weight>")
    
    upper = (datetime.datetime.strptime(REF_DATE,'%Y/%m/%dT%H:%M:%S').date() - \
             pd.to_datetime(df_usage['Date'], format="%Y/%m/%dT%H:%M:%S")).dt.total_seconds()

    if(weights):
        affinity = df_usage['Weight']*np.exp(-1.0*upper/3600.0/24.0/DECAY)
    else:
        affinity = np.exp(-1.0*upper/3600.0/24.0/DECAY)
    affinity_frame = pd.concat([df_usage['UserId'], df_usage['ItemId'], affinity], axis =1)
    affinity_frame.columns = ['UserId', 'ItemId', 'affinity']
    # sum affinity if the userID and itemID are same.
    df = affinity_frame.groupby(['UserId', 'ItemId'],as_index = False)['affinity'].sum()

    # remove outliers
    #If the value is larger than the upper threshold, it will be set as the upper threshold
    p = np.percentile(df['affinity'], p)
    df['affinity'] = np.minimum(df['affinity'], p).tolist()

    print("DEBUG: Affinity Cap:", p)
    print("DEBUG: Affinity shape:", df.shape)

    return df