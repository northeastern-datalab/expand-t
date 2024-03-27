import time
import os
import glob
import pandas as pd
import argparse
import json
from discover_candidates import CandidateTables
from tqdm import tqdm

def get_lake(benchmark):
    '''
    Get data lake tables found from Starmie
    Args: 
        benchmark(str): benchmark name for filepath
        sourceTableName (str): name of source table
        discludeTables (list): tables to disclude from data lake tables (source)
        includeStarmie (Boolean): get candidate tables from Starmie results or not
    Return: 
        lakeDfs(dict of dicts): filename: {col: list of values as strings}
        rawLakeDfs (dict): filename: raw DataFrame
    '''
    DATALAKE_PATH = '/home/gfan/Datasets/%s/' % (benchmark)
    # Import all data lake tables
    totalLakeTables = glob.glob(DATALAKE_PATH+'datalake/*.csv')
    
    rawLakeDfs = {}
    allLakeTableCols = {}
    for filename in totalLakeTables:
        table = filename.split("/")[-1]
        df = pd.read_csv(filename, lineterminator="\n")
        rawLakeDfs[table] = df
        
        for index, col in enumerate(df.columns):    
            if table not in allLakeTableCols: allLakeTableCols[table] = {}
            # Convert every data value to strings, so there are no mismatches from data types
            allLakeTableCols[table][col] = [str(val).rstrip() for val in df[col] if not pd.isna(val)]
    return rawLakeDfs, allLakeTableCols

def get_starmie_candidates(benchmark):
    '''
    Get data lake tables found from Starmie
    Args: 
        benchmark(str): benchmark name for filepath
    Return: 
        starmieCandidatesForSources(dict): source table: list of candidate tables returned from Starmie
    '''
    # ==== Import the tables returned from Starmie and use that as reduced data lake
    with open("../../Starmie_candidate_results/%s/starmie_candidates.json" % (benchmark)) as json_file: starmieCandidatesForSources = json.load(json_file)
    return starmieCandidatesForSources