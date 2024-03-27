import sys
import numpy as np
import pandas as pd
import time

replaceNull = '*NAN*'

def FindCurrentNullPattern(tuple1):
    current_pattern = ""
    current_nulls = 0
    for t in tuple1:
        if pd.isna(t):
            current_pattern += "0"
            current_nulls += 1
        else:
            current_pattern += "1"
    return current_pattern, current_nulls

#used to check what are the ancestor buckets of the child bucket
def CheckAncestor(child_bucket, parent_bucket):
    for i in range(len(child_bucket)):
        if int(child_bucket[i]) == 1 and int(parent_bucket[i])==0:
            return 0
    return 1

def CheckNonNullPositions(tuple1, total_non_nulls):
    non_null_positions = set()
    for i in range(0, len(tuple1)):
        if int(tuple1[i]) == 1:
            non_null_positions.add(i)
            if len(non_null_positions) == total_non_nulls:
                return non_null_positions
    return (non_null_positions)

def GetProjectedTuple(tuple1, non_null_positions, m):
    projected_tuple = tuple()
    for j in range(0,m):
        if j in non_null_positions:
            projected_tuple += (tuple1[j],)
    return projected_tuple

def labelNullsDf(queryDf, df, keyCols):
    try: 
        queryDf = queryDf[df.columns]
    except: return None
    labeledNullsDf = df.copy()
    originalCols = labeledNullsDf.columns
    dfKeyCols = [col for col in keyCols if col in originalCols]
    if not dfKeyCols: return None
    keyCol = dfKeyCols[0]
    for keyVal in queryDf[keyCol]:
        if pd.isna(keyVal) and not [val for val in labeledNullsDf[keyCol].values if pd.isna(val) or val == replaceNull]:
            nullTuple = [[np.nan]*len(df.columns)]
            labeledNullsDfVals = labeledNullsDf.values.tolist()
            labeledNullsDf = pd.DataFrame(labeledNullsDfVals+nullTuple, columns=originalCols)
            df = labeledNullsDf
        qTuples = queryDf.loc[queryDf[keyCol]==keyVal].values.tolist()
        candTuples = df.loc[df[keyCol]==keyVal].values.tolist()
        candIndxes = df.index[df[keyCol]==keyVal].tolist()
        if pd.isna(keyVal):
            qTuples = queryDf[queryDf[keyCol].isnull()].drop_duplicates().values.tolist()
            candTuples = df[df[keyCol].isnull()].drop_duplicates().values.tolist()
            candIndxes = df.index[df[keyCol].isnull()].drop_duplicates().values.tolist()
        qTuple = qTuples[0]
        qTupleNullIndxs = [i for i, val in enumerate(qTuple) if pd.isna(val)]
        for cInd, cT in enumerate(candTuples):
            qTupleNullIndxs = [i for i, val in enumerate(qTuple) if pd.isna(val)]
            candTupleNullIndxs = [i for i, cval in enumerate(cT) if pd.isna(cval)]
            commonNullIndxs = list(set(qTupleNullIndxs).intersection(set(candTupleNullIndxs)))
            for colInd in commonNullIndxs:
                labeledNullsDf.loc[candIndxes[cInd], originalCols[colInd]] = replaceNull
    labeledNullsDf.columns = originalCols
    return labeledNullsDf

def ReplaceNulls(table, null_count):
    null_set = set()
    for colname in table:
        for i in range (0, table.shape[0]):
            try:
                if pd.isna(table[colname][i]):
                    table[colname][i] = "null"+ str(null_count)
                    null_set.add("null"+ str(null_count))
                    null_count += 1
            except:
                sys.exit()
    return table, null_count, null_set

def AddNullsBack(table, nulls):
    columns = list(table.columns)
    input_rows = list(tuple(x) for x in table.values)
    output_rows = []
    for t in input_rows:
        new_t = tuple()
        for i in range(0, len(t)):
            if str(t[i]) in nulls:
                new_t += (np.nan,)
            else:
                new_t += (t[i],)
        output_rows.append(new_t)
    final_table = pd.DataFrame(output_rows, columns =columns)
    return final_table

# =============================================================================
# Efficient complementation using partitioning starts here
# =============================================================================
def complementTuples(tuple1, tuple2):
    keys = 0 #find if we have common keys
    alternate1= 0 #find if we have alternate null position with non-null value in the first tuple
    alternate2 = 0 #find if we have alternate null position with non-null value in the second tuple
    newTuple = list()
    
    for i in range(0,len(tuple1)):
        first = tuple1[i]
        if pd.isna(first): first = "nan"
        second = tuple2[i]
        if pd.isna(second): second = "nan"
        if first != "nan" and second!="nan" and first != second:
            return (tuple1,False)
        elif first == "nan" and second =="nan":
            # newTuple.append(first)
            newTuple.append(np.nan)
            
        elif first != "nan" and second!="nan" and first == second: #both values are equal
            keys+=1
            newTuple.append(first)
        #second has value and first is null
        elif first == "nan" and second != "nan":
            alternate1+=1
            newTuple.append(second)
        #first has value and second is null
        elif (second =="nan" and first != "nan"):
            alternate2+=1
            newTuple.append(first)
    count = 0
    for item in newTuple:
        if(pd.isna(item)):    
            count+=1     
    if (keys >0 and alternate1 > 0 and alternate2>0 and count != len(newTuple)):
        return (tuple(newTuple),True)
    else:
        return (tuple(tuple1),False)
    

        
def PartitionTuples(table, partitioning_index):
    partitioned_tuple_dict = dict()
    all_tuples = [tuple(x) for x in table.values]
    for t in all_tuples:
        if t[partitioning_index] in partitioned_tuple_dict:
            partitioned_tuple_dict[t[partitioning_index]].append(t)
        else:
            partitioned_tuple_dict[t[partitioning_index]] = [t]
    return partitioned_tuple_dict

def GetPartitionsFromList(all_tuples, partitioning_index):
    partitioned_tuple_dict = dict()
    for t in all_tuples:
        if t[partitioning_index] in partitioned_tuple_dict:
            partitioned_tuple_dict[t[partitioning_index]].add(t)
        else:
            partitioned_tuple_dict[t[partitioning_index]] = {t}
    null_partition = partitioned_tuple_dict.pop(np.nan, None)
    if null_partition is None:
        for each in partitioned_tuple_dict:
            partitioned_tuple_dict[each] = list(partitioned_tuple_dict[each])
        return partitioned_tuple_dict
    else:
        if len(partitioned_tuple_dict) == 0:
            partitioned_tuple_dict[np.nan] = list(null_partition)
            return partitioned_tuple_dict
        for each in partitioned_tuple_dict:
            temp_list = partitioned_tuple_dict[each]
            temp_list = temp_list.union(null_partition)
            partitioned_tuple_dict[each] = list(temp_list)            
    return partitioned_tuple_dict

def SelectPartitioningOrder(table):
    statistics = dict()
    stat_unique = {}
    stat_nulls = {}
    total_rows = table.shape[0]
    unique_weight = 0
    null_weight = 1 - unique_weight #only based on null weight
    i = 0
    for col in table:
        unique_count = len(set(table[col]))
        null_count = total_rows - table[col].isna().sum()
        score = (unique_count * unique_weight) + null_count * null_weight
        statistics[i] = score
        stat_unique[i] = unique_count
        stat_nulls[i] = total_rows - null_count
        i += 1
    stat_nulls = sorted(stat_nulls, key = stat_nulls.get, reverse = True)
    stat_unique = sorted(stat_unique, key = stat_unique.get, reverse = True)
    final_list = [stat_nulls[0]]
    stat_unique.remove(stat_nulls[0])
    final_list += stat_unique
    #return final_list    
    return sorted(statistics, key = statistics.get, reverse = True)

def FineGrainPartitionTuples(table, timeout, fdaStartTime):  
    input_tuples = list({tuple(x) for x in table.values})
    partitioning_order = SelectPartitioningOrder(table)
    debug_dict = {}
    list_of_list = []
    assign_tuple_id = {}
    for tid, each_tuple in enumerate(input_tuples):
        assign_tuple_id[each_tuple] = tid 
        if (time.time() - fdaStartTime) > timeout: return None, None
    list_of_list.append(input_tuples)
    finalized_list = []
    for i in partitioning_order:
        new_tuples = []
        track_used_tuples = {}
        for all_tuples in list_of_list:
            if len(all_tuples) > 100:
                partitions = GetPartitionsFromList(all_tuples, i)
                for each in partitions:
                    current_partition = partitions[each]
                    create_tid = set()
                    for current_tuple in current_partition:
                        create_tid.add(assign_tuple_id[current_tuple])
                    create_tid = tuple(sorted(create_tid))
                    if create_tid not in track_used_tuples:
                        if len(current_partition) > 100:
                            new_tuples.append(current_partition)
                        else:
                            finalized_list.append(current_partition)
                        track_used_tuples[create_tid] = 1
            else:
                finalized_list.append(all_tuples)
            if (time.time() - fdaStartTime) > timeout: return None, None
        list_of_list = new_tuples
        debug_dict[i] = list_of_list
    if len(list_of_list) > 0:    
        finalized_list = list_of_list + finalized_list
    return finalized_list, debug_dict


def ComplementAlgorithm(tuple_list, timeout, fdaStartTime):
    receivedTuples = dict()
    for t in tuple_list:
        receivedTuples[t] = 1
    complementResults = dict()
    while (1):
        i = 1
        used_tuples = dict()
        for tuple1 in tuple_list:
            complementCount = 0
            for tuple2 in tuple_list[i:]:
                (t, flag) = complementTuples(tuple1, tuple2)
                if (flag == True):
                    complementCount += 1
                    complementResults[t] = 1
                    used_tuples[tuple2] = 1
            i += 1
            if complementCount == 0 and tuple1 not in used_tuples:
                complementResults[tuple1] = 1
            if (time.time() - fdaStartTime) > timeout: return None
        if receivedTuples.keys() == complementResults.keys():
            break
        else:
            receivedTuples = complementResults
            complementResults = dict()
            tuple_list = [tuple(x) for x in receivedTuples]
    return [tuple(x) for x in complementResults]
    

# =============================================================================
# Efficient complementation using partitioning ends here
# =============================================================================


def innerUnion(tableDfs, primaryKey, foreignKeys):
    '''
    Directly union tables that have the same schemas
    '''
    innerUnionOp = []
    unionableTables = {}
    for tableA in tableDfs:
        if tableA in sorted({x for v in unionableTables.values() for x in v}):
            continue
        ASchema = list(tableDfs[tableA].columns)
        for tableB in tableDfs:
            if tableA != tableB:
                BSchema = list(tableDfs[tableB].columns)
                commonCols = set(ASchema).intersection(set(BSchema))
                if commonCols == set(ASchema) and commonCols == set(BSchema):
                    if tableA not in unionableTables:
                        unionableTables[tableA] = [tableB]
                    else:
                        unionableTables[tableA].append(tableB)
    delTables = set()
    
    for tableA in unionableTables:
        null_count = 0
        null_set = set()
        delTables.add(tableA)
        tableADf = tableDfs[tableA]
        union_df, union_name  = pd.DataFrame(), ""
        canUnion = True
        for tableB in unionableTables[tableA]:
            tableBDf = tableDfs[tableB][tableADf.columns]
            delTables.add(tableB)
            # iterate through aligned schemas and union
            union_df = pd.concat([tableADf, tableBDf], ignore_index=True)  
            innerUnionOp.append(f"inner_union:{tableA},{tableB}")
            union_name = tableA.split(".csv")[0] + "," + tableB
            tableA, tableADf = union_name, union_df
        tableDfs[union_name] = union_df
    for table in delTables:
        del tableDfs[table]
        
    # maintain order of tables in tableDfs
    orderedTables = [list(tableDfs.keys())[0]]
    for tableIndx, (table, df) in enumerate(tableDfs.items()):
        if tableIndx == (len(tableDfs)-1): 
            if table not in orderedTables: orderedTables.append(table)
            break
        nextTable = list(tableDfs.keys())[tableIndx+1]
        nextDf = list(tableDfs.values())[tableIndx+1]
        commonCols = [col for col in df.columns if col in nextDf.columns]
        if len(commonCols) > 0: orderedTables.append(nextTable)
        else:
            for nextTableIndx, (nextTable, nextDf) in enumerate(tableDfs.items()):
                if nextTableIndx <= (tableIndx+1): continue
                commonCols = [col for col in df.columns if col in nextDf.columns]
                if len(commonCols) > 0: 
                    orderedTables.append(nextTable)
                    # swap positions in tableDfs
                    allTables = list(tableDfs.items())
                    allTables[tableIndx+1], allTables[nextTableIndx] = allTables[nextTableIndx], allTables[tableIndx+1]
                    tableDfs = dict(allTables)
                    break
    orderedTableDfs = {table: tableDfs[table] for table in orderedTables}
    return orderedTableDfs, innerUnionOp

def projectAtts(tableDfs, queryTable):
    ''' For each table, project out the attributes that are in the query table.
        If the table has no shared attributes with the query table, remove
    '''
    project_ops = []
    projectedDfs = {}
    queryCols = queryTable.columns
    for table, df in tableDfs.items():
        projectedDfs[table] = df
        tableCols = df.columns
        projectedTable = df.drop(columns=[c for c in tableCols if c not in queryCols])
        if not projectedTable.empty:
            projectedDfs[table] = projectedTable
            if projectedTable.shape[1] != df.shape[1]: project_ops.append(f"projection:{table}")
    return projectedDfs,project_ops

def selectKeys(tableDfs, queryTable, primaryKey, foreignKeys):
    ''' For each table, select tuples that contain key value from queryTable
        If the table has no shared keys with the queryTable, remove
    '''    
    select_ops = []
    selectedDfs = {}
    queryKeyVals = {}
    for col in queryTable.columns:
        queryKeyVals[col] = queryTable[col].tolist()
        
    commonKey = primaryKey
    for table, df in tableDfs.items():
        selectedDfs[table] = df.drop_duplicates().reset_index(drop=True)
        
        dfCols = df.columns
        commonKeys = [k for k in [primaryKey]+foreignKeys if k in dfCols]
        allColNumVals = {}
        commonKey = None
        if commonKeys: 
            commonKey = commonKeys[0]
            numCommonKeyUniqueVals = len(set([val for val in df[commonKey].values.tolist() if not pd.isna(val)]))
        
        for col in dfCols:
            uniqueVals = set([val for val in df[col].values.tolist() if not pd.isna(val)])            
            if col in commonKeys and col != commonKey:
                if len(uniqueVals) > numCommonKeyUniqueVals:
                    numCommonKeyUniqueVals = len(uniqueVals)                    
                    commonKey = col
            elif col not in commonKeys: allColNumVals[col] = len(uniqueVals)
        allColNumVals = {k: v for k, v in sorted(allColNumVals.items(), key=lambda item: item[1], reverse=True)}
        tableFK = list(allColNumVals.keys())
        if commonKey:
            commonKeys = [commonKey]
            if tableFK: commonKeys.append(tableFK[0])
            conditions = [df[commonKeys[0]].isin(queryKeyVals[commonKeys[0]]).values]
            if len(commonKeys) > 1:
                for commonKey in commonKeys[1:]:
                    conditions.append([df[commonKey].isin(queryKeyVals[commonKey]).values])    
            conditions.append(np.full((1,len(conditions[0])), True, dtype=bool))              
            selectedTuples = df.loc[np.bitwise_and.reduce(conditions)[0]]
        else: 
            commonKeys = tableFK[:2]
            print("%s commonKeys: " % (table), commonKeys)
            conditions = [df[commonKeys[0]].isin(queryKeyVals[commonKeys[0]]).values]
            if len(commonKeys) > 1:
                for commonKey in commonKeys[1:]:
                    conditions.append([df[commonKey].isin(queryKeyVals[commonKey]).values])                
            else: conditions.append(np.full((1,len(conditions[0])), False, dtype=bool))  
            selectedTuples = df.loc[np.bitwise_or.reduce(conditions)[0]]
             
        if not selectedTuples.empty:
            selectedDfs[table] = selectedTuples 
            if selectedTuples.shape[0] != df.shape[0]: select_ops.append(f"selection:{table}")
    return selectedDfs,select_ops

# =============================================================================
# Running Outerjoin as baseline
# =============================================================================
def outerjoin(dfs_list):
    '''
    Given a list of DFs, perform outerjoin in order of the list
    '''
    if not dfs_list:
        return None
    # Start with the first DataFrame in the list
    merged_df = dfs_list[0]
    # Iterate over the remaining DataFrames and merge them
    for df in dfs_list[1:]:
        # Find common columns for merging
        common_columns = list(set(merged_df.columns) & set(df.columns))
        # Perform outer join using common columns
        merged_df = pd.merge(merged_df, df, on=common_columns, how='outer')
    return merged_df
    

def detect_sensitive_att(column):
    '''
    Detect if the column is a sensitive attribute in https://www.justice.gov/crt/fair-housing-act-1
    '''
    isSensitiveAtt = False
    sensitive_attributes = ["race", "color", "religion", "sex", "nationality", "familial status", "status", "disability"]
    if column.lower() in sensitive_attributes:
        isSensitiveAtt = True
    return isSensitiveAtt

def truncate_float_vals(source_df, df):
    truncate_float_vals = {}
    # Select only float64 columns
    float_columns = df.select_dtypes(include='float64')
    # Iterate through each float64 column
    for column in float_columns.columns:
        if column in source_df and source_df[column].dtype == 'float64': continue
        truncate_float_vals[column] = "{:.0f}"
    return truncate_float_vals