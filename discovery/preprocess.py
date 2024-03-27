import pandas as pd
import os
import json
import sys
sys.path.append('../')

def preprocessForeignKeys(tableDfs, primaryKey, foreignKeys, sourceTable):
    '''
    First Join tables with foreign keys to those with primary keys
    '''
    tablesToForeignJoin = set()
    foreignJoinedTableDfs = tableDfs.copy()
    for tableA, dfA in tableDfs.items():
        dfASchema = dfA.columns.tolist()
        if [col for col in [primaryKey]+foreignKeys if col in dfASchema]:
            # have overlapping key columns with source table
            continue
        tablesToForeignJoin.add(tableA)
        foundKeyPath = 0
        joinTable = None, None
        candidateJoinTableWithKey, candidateJoinTables = {}, {} # candidateJoinTable: additional Source table columns it contains
        while not foundKeyPath:
            for tableB, dfB in tableDfs.items():
                if tableB == tableA: continue
                dfBSchema = dfB.columns.tolist()
                commonCols = set(dfASchema).intersection(set(dfBSchema))
                if not commonCols or set(dfASchema) == set(dfBSchema):continue
                remainSourceCols = [col for col in dfB if col not in commonCols and col in sourceTable]
                if [col for col in [primaryKey]+foreignKeys if col in remainSourceCols]: 
                    foundKeyPath = 1
                for cCol in commonCols:
                    AColVals = dfA[cCol].values
                    BColVals = dfB[cCol].values
                    commonColVals = set(AColVals).intersection(set(BColVals))
                    if not commonColVals: continue
                    if tableB not in candidateJoinTableWithKey: candidateJoinTableWithKey[tableB] = []
                    if foundKeyPath:
                        candidateJoinTableWithKey[tableB] = remainSourceCols
                    else: candidateJoinTables[tableB] = remainSourceCols
                    break
            if candidateJoinTableWithKey:
                candidateJoinTableWithKey = {k: v for k, v in sorted(candidateJoinTableWithKey.items(), key=lambda item: len(item[1]), reverse=True)}
                joinTable = list(candidateJoinTableWithKey.keys())[0]
            else:
                candidateJoinTables = {k: v for k, v in sorted(candidateJoinTables.items(), key=lambda item: len(item[1]), reverse=True)}
                joinTable = list(candidateJoinTables.keys())[0]
            
            if joinTable: tablesToForeignJoin.remove(tableA)
            outer_join_df = pd.merge(dfA, foreignJoinedTableDfs[joinTable], how='outer')
            foreignJoinedTableDfs[tableA] = outer_join_df
            foundKeyPath = 1
    if tablesToForeignJoin:
        for table in tablesToForeignJoin:
            del foreignJoinedTableDfs[table]
    return foreignJoinedTableDfs
        
        
def getTableDfs(benchmark, datasets, sourceTableName, candidates={}):
    candidateTableDict = candidates
    if not candidates:
        # saved to hard drive
        candidateTablePath = "../results_candidate_tables/%s/candidateTables.json" % (benchmark)
        
        if not os.path.isfile(candidateTablePath): return None
        with open(candidateTablePath) as json_file: candidateTableDict = json.load(json_file)[sourceTableName]
    tableDfs = {}
    numAdditional = 0
    for table in datasets:
        tableName = table.split("/")[-1]
        if tableName in candidateTableDict: 
            if tableName == sourceTableName: continue
            table_df = pd.read_csv(table)
            table_df = table_df.rename(columns=candidateTableDict[tableName])
            tableDfs[tableName] = table_df
                        
    return tableDfs
