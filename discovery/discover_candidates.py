import pandas as pd
import numpy as np
import time
from pandas.api.types import is_numeric_dtype
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)    
pd.options.mode.chained_assignment = None

class CandidateTables:
    def __init__(self, benchmark, rawLakeDfs, allLakeTableCols, starmie_candidates):
        self.benchmark = benchmark
        self.sim_threshold = 0.2
        self.rawLakeDfs = rawLakeDfs
        self.lakeDfs = {}
        self.allLakeTableCols = allLakeTableCols
        self.starmie_candidates = starmie_candidates
        
        self.source_df = pd.DataFrame()
        self.candidateTablesFound = {} # data lake table: {related dl table column: source table column}
        self.noCandidates = 0 # flag: no candidates found for this Source Table

    def get_set_overlap(self, sourceCol):
        '''
        Get tables with columns that have a set overlap with current source column above threshold
        Args: 
            sourceCol (pd.Series): current column in the Source Table that we use to query
        '''
        all_similarities = {} # table: overlap score > threshold
        similar_columns = {} # table: (mostCommonCol, list of values of most common column with source)
        # Convert every data value to strings, so there are no mismatches from data types
        source_col_set = set([str(q).rstrip() for q in sourceCol if not pd.isna(q)])
        # compute Set Overlap
        for table in self.lakeDfs:
            # for each table, find a column that contains the highest set overlap with current source col, and check if > threshold
            max_ind_overlap = 0.0
            for col in self.lakeDfs[table]: # for each column in current data lake table
                col_vals = self.lakeDfs[table][col] # list of values
                overlap = len(list(source_col_set.intersection(set(col_vals)))) # set intersection
                
                if overlap > max_ind_overlap:
                    max_ind_overlap = overlap
                    similar_columns[table] = (col, col_vals)
            # Only add if the overlap / length of source set is greater than threshold
            if (max_ind_overlap / len(source_col_set)) >= self.sim_threshold:
                # if col_vals and (len(set(col_vals)) - overlap)/len(set(col_vals)) < (overlap)/len(set(col_vals)): # majority is not noise
                all_similarities[table] = (max_ind_overlap / len(source_col_set))
        # Sort similarities by similarity score in decreasing order
        all_similarities = {k: v for k, v in sorted(all_similarities.items(), key=lambda item: item[1], reverse=True)}
        return all_similarities, similar_columns

    def get_diverse_set_overlap(self, sourceCol):
        '''
        MAXIMIZE similarity with sourceCol while MINIMIZING similarity with previous table
        Arguments:
            sourceCol (pd.Series) = current column in source
        Return 
            diverse_table_similarities (dict): table: (column that has high overlap with sourceCol, score)
            table_source_similarities (dict): returned from get_set_overlap (table: overlap score)
        '''
        # table_source_similarities (dict): DL table: maximum individual overlap
        # similar_df_columns (dict): DL table: (1 most common col, column values)
        table_source_similarities, similar_df_columns = self.get_set_overlap(sourceCol)
        diverse_table_similarities = {}
        for tableIndx, (currSimTable, tableScore) in enumerate(table_source_similarities.items()):
            diverse_score = tableScore
            currTableCol = similar_df_columns[currSimTable][1]
            if tableIndx > 0: 
                prevSimTable = list(table_source_similarities.keys())[tableIndx-1]
                prevTableCol = similar_df_columns[prevSimTable][1]
                # sim(currTable, sourceTable) - sim(currTable, prevTable)
                prev_curr_overlap = len(list(set(currTableCol).intersection(set(prevTableCol))))
                diverse_score -= (prev_curr_overlap / len(set(currTableCol)))
            diverse_table_similarities[currSimTable] = (similar_df_columns[currSimTable][0], diverse_score)
        diverse_table_similarities = {k: v for k, v in sorted(diverse_table_similarities.items(), key=lambda item: item[1][1], reverse=True)}
        return diverse_table_similarities, table_source_similarities

    def check_subsumed_tables(self, tables_aligned_tuple_indxes):
        '''
        Remove tables whose values are covered by another table -- ensure that all overlapping values are in sourceTable
        similar_tableCols[table] = [(table_similar_col, sCol)]
        '''
        sourceCols = self.source_df.columns.tolist()
        table_alignedCols = self.candidateTablesFound.copy()
        table_alignedDfs = {}
        removeTables = set()
        for table in table_alignedCols:
            df = self.rawLakeDfs[table]
            df = self.rawLakeDfs[table].rename(columns=table_alignedCols[table])
            df = df[[col for col in sourceCols if col in list(table_alignedCols[table].values())]]
            dfCols = list(df.columns)
            
            if len(dfCols) > 1:
                conditions = [df[dfCols[0]].isin(self.source_df[dfCols[0]]).values]
                for col in dfCols[1:]:
                    conditions.append([df[col].isin(self.source_df[col]).values])       
                df = df.loc[np.bitwise_or.reduce(conditions, dtype=object)[0]].drop_duplicates().reset_index(drop=True)
                
            else:
                df = df[df[dfCols[0]].isin(self.source_df[dfCols[0]])].drop_duplicates().reset_index(drop=True)
                        
            if not df.empty:
                table_alignedDfs[table] = df
        for tableA, dfA in table_alignedDfs.items():
            if tableA in removeTables: continue
            ARows = [tuple(row) for row in dfA.values]
            ASourceColNames = set(dfA.columns)
            for tableB, dfB in table_alignedDfs.items():
                if tableB in removeTables or tableA in removeTables: continue
                if tableA == tableB or dfA.equals(dfB): continue
                BRows = [tuple(row) for row in dfB.values]
                BSourceColNames = set(dfB.columns)
                subsumed, subsumedRows, subsumer, subsumerRows = None, None, None, None
                if len(ASourceColNames) < len(BSourceColNames) and ASourceColNames.issubset(BSourceColNames):
                    subsumed, subsumer = tableA, tableB
                    subsumedRows, subsumerRows = ARows, BRows
                elif len(BSourceColNames) < len(ASourceColNames) and BSourceColNames.issubset(ASourceColNames):
                    subsumed, subsumer = tableB, tableA
                    subsumedRows, subsumerRows = BRows, ARows
                elif ASourceColNames == BSourceColNames:
                    overlapRows = set(ARows).intersection(set(BRows))
                    onlyARows = [row for row in ARows if row not in overlapRows]
                    onlyBRows = [row for row in BRows if row not in overlapRows]
                    if not onlyARows and not onlyBRows: continue
                    if not onlyARows: 
                        removeTables.add(tableA)
                    if not onlyBRows:
                        removeTables.add(tableB)
                    continue
                if subsumed == None: continue
                sharedCols = set([col for col in ASourceColNames if col in BSourceColNames])
                
                if not sharedCols: continue
                subsumedTuples = []
                
                for row1 in set(subsumedRows):
                    for row2 in set(subsumerRows):
                        overlapValsInRow = tuple([val for val in row1 if val in row2])
                        if overlapValsInRow == row1:
                            subsumedTuples.append(row1)
                            break
                        elif overlapValsInRow == row2:
                            subsumedTuples.append(row2)
                            break
                if set(subsumedTuples) == set(subsumedRows) and set(subsumedTuples) == set(subsumerRows): continue
                if set(subsumedTuples) == set(subsumedRows):
                    removeTables.add(subsumed)
                elif set(subsumedTuples) == set(subsumerRows):
                    removeTables.add(subsumer)
        for table in removeTables:
            del self.candidateTablesFound[table]

    def get_lake(self,sourceTableName):
        '''
        Get data lake tables found from Starmie
        Return: 
            lakeDfs(dict of dicts): filename: {col: list of values as strings}
        '''
        retrievedTables = list(self.rawLakeDfs.keys())
        # ==== Import the tables returned from Starmie and use that as reduced data lake
        if self.starmie_candidates: retrievedTables = self.starmie_candidates
        for filename in retrievedTables:
            table = filename.split("/")[-1]
            if self.benchmark == 't2d_gold' and sourceTableName.replace("_1.csv", ".csv") == table: continue
            self.lakeDfs[table] = self.allLakeTableCols[table]

    def find_candidates(self, sourceTableName):
        '''
        Get Tables whose columns have high set overlap with Source Table's columns
        Args:
            sourceTableName: name of the Source Table
            includeStarmie: discover candidate tables from tables returned using Starmie
        Save candidate tables (and their aligned columns to Source Table's columns to rename to) to a pkl file
        '''
        source_PATH = "/home/gfan/Datasets/%s/queries/%s" % (self.benchmark,sourceTableName)
        self.source_df = pd.read_csv(source_PATH, lineterminator="\n")
        
        self.get_lake(sourceTableName)
    
        source_times = []
        similar_tableCols = {}
        tableDiverseSimScores, tableSetSimScores = {}, {}
        sourceCols = list(self.source_df.columns)
        primaryKey = self.source_df.columns.tolist()[0]
        
        for sCol in sourceCols:
            source_col = sCol.rstrip()
            source_start_time = time.time()
            source_set = set([str(q).rstrip() for q in self.source_df[sCol] if not pd.isna(q)])
            if not source_set: 
                print(f"Source Column {sCol} is empty")
                continue
            # Overlap similarity
            diverse_res, set_overlap_res = self.get_diverse_set_overlap(self.source_df[sCol])
            if len(diverse_res) == 0: continue
            if sCol == primaryKey and len(diverse_res) == 0: 
                # no candidate tables contain columns that share values with the primary key -> exit
                self.noCandidates = 1
                break
            
            source_times.append(time.time() - source_start_time)
            # Get inverse dictionary: the source columns for which the table is found
            for table in diverse_res:
                table_similar_col = diverse_res[table][0]
                col_diverse_sim_score = diverse_res[table][1]
                col_set_sim_scores = set_overlap_res[table]
                
                if table in tableDiverseSimScores: 
                    tableDiverseSimScores[table].append(col_diverse_sim_score)
                    tableSetSimScores[table].append(col_set_sim_scores)
                else: 
                    tableDiverseSimScores[table] = [col_diverse_sim_score]
                    tableSetSimScores[table] = [col_set_sim_scores]
                if table in similar_tableCols: similar_tableCols[table].append((table_similar_col,  sCol))
                else: similar_tableCols[table] = [(table_similar_col, sCol)]
        
        tableDiverseSimScores = {k: sum(v)/len(v) for k, v in tableDiverseSimScores.items()}
        tableDiverseSimScores = {k: v for k, v in sorted(tableDiverseSimScores.items(), key=lambda item: item[1], reverse=True)}
        
        tableSetSimScores = {k: sum(v)/len(v) for k, v in tableSetSimScores.items() if sum(v)/len(v) >= self.sim_threshold}
        # Rename columns of data lake tables to match source columns
        tables_aligned_tuple_indxes = {}
        for dl_table in tableDiverseSimScores:   
            if dl_table not in tableSetSimScores: continue 
            dl_table_schema = list(self.lakeDfs[dl_table].keys())
            if len(similar_tableCols[dl_table]) > 1:
                if len(set(dl_table_schema).intersection(set(sourceCols))) > 1:
                    self.candidateTablesFound[dl_table] = {}
                    for col in set(dl_table_schema).intersection(set(sourceCols)):
                        self.candidateTablesFound[dl_table][col] = col
                    continue
                # get number of non-numeric columns
                non_numerical_similar_qCols = [similar_cols[0] for similar_cols in similar_tableCols[dl_table] if not is_numeric_dtype(self.rawLakeDfs[dl_table][similar_cols[0]])]
                if len(non_numerical_similar_qCols) > 0:
                    # if there are any non-numeric columns in common (disregarding numeric keys)
                    self.candidateTablesFound[dl_table] = {}
                    dl_table_cols = [similar_cols[0] for similar_cols in similar_tableCols[dl_table]]
                    overlapAlignedColIndxes = {}
                    for similar_cols in similar_tableCols[dl_table]: 
                        dlSimilarCol = similar_cols[0]
                        source_SimilarCol = similar_cols[1]
                        currDlTable = self.rawLakeDfs[dl_table]
                        currDlCol = currDlTable[dlSimilarCol]
                        currDlTable.reset_index(drop=True, inplace=True)
                        # align tuples and check if the columns that independently have high overlap with a source table's column still have high overlap within the aligned tuples
                        if dlSimilarCol in overlapAlignedColIndxes: overlapAlignedColIndxes[dlSimilarCol][source_SimilarCol] = []
                        else: overlapAlignedColIndxes[dlSimilarCol] = {source_SimilarCol: []}
                        for valIndx, val in enumerate(currDlCol.values.tolist()):
                            if val in self.source_df[source_SimilarCol].values.tolist():
                                overlapAlignedColIndxes[dlSimilarCol][source_SimilarCol].append(valIndx)
                    overlapColIndxes = {}
                    foundIndxes = set()
                    overlapIndxes = set()
                    allAlignedColsIndxes = overlapAlignedColIndxes.copy()
                    for dlCol in allAlignedColsIndxes:
                        allMappedSourceCols = list(overlapAlignedColIndxes[dlCol].keys())
                        for sCol in allMappedSourceCols:
                            if sCol in overlapAlignedColIndxes[dlCol] and not overlapAlignedColIndxes[dlCol][sCol]: 
                                del overlapAlignedColIndxes[dlCol][sCol]
                                continue
                        if len(overlapAlignedColIndxes[dlCol]) > 1:
                            # same data lake column maps to multiple source columns
                            keepSourceCol = max(overlapAlignedColIndxes[dlCol], key= lambda x: len(set(overlapAlignedColIndxes[dlCol])))
                            allMappedSourceCols = list(overlapAlignedColIndxes[dlCol].keys())
                            for sCol in allMappedSourceCols:
                                if sCol != keepSourceCol:
                                    del overlapAlignedColIndxes[dlCol][sCol]
                            if keepSourceCol in overlapAlignedColIndxes[dlCol] and overlapAlignedColIndxes[dlCol][keepSourceCol]: 
                                overlapColIndxes[dlCol] = overlapAlignedColIndxes[dlCol][keepSourceCol]
                        elif len(overlapAlignedColIndxes[dlCol]) == 1:
                            overlapColIndxes[dlCol] = list(overlapAlignedColIndxes[dlCol].values())[0]
                        if not overlapAlignedColIndxes[dlCol]: 
                            del overlapAlignedColIndxes[dlCol]
                        if dlCol in overlapColIndxes:
                            for ind in overlapColIndxes[dlCol]:
                                if ind in foundIndxes:
                                    overlapIndxes.add(ind)
                                else: foundIndxes.add(ind)
                            
                        
                    tables_aligned_tuple_indxes[dl_table] = overlapIndxes
                    for dlSimilarCol in overlapAlignedColIndxes:
                        source_SimilarCol = list(overlapAlignedColIndxes[dlSimilarCol].keys())[0]
                        # check if there is still set similarity after alignment
                        valOverlap = [val for val in set(self.rawLakeDfs[dl_table][dlSimilarCol].values.tolist()) if val in set(self.source_df[source_SimilarCol].values.tolist())]
                        
                        alignedValOverlap = [val for i, val in enumerate(self.rawLakeDfs[dl_table][dlSimilarCol].values.tolist()) if i in overlapIndxes]
                        
                        if not valOverlap or not alignedValOverlap: 
                            dl_table_cols.remove(dlSimilarCol)
                            continue
                        if (len(set(valOverlap)) / len(set(self.source_df[source_SimilarCol].values))) < self.sim_threshold: 
                            dl_table_cols.remove(dlSimilarCol)
                            continue
                        if (len(set(alignedValOverlap)) / len(set(self.source_df[source_SimilarCol].values))) < self.sim_threshold: 
                            dl_table_cols.remove(dlSimilarCol)
                            continue
                        
                        self.candidateTablesFound[dl_table][dlSimilarCol] = source_SimilarCol
                    non_numerical_similar_qCols = [col for col in dl_table_cols if not is_numeric_dtype(self.rawLakeDfs[dl_table][col])]
                    if not non_numerical_similar_qCols or not self.candidateTablesFound[dl_table]:
                        self.candidateTablesFound.pop(dl_table)
        if len(self.candidateTablesFound) == 0: self.noCandidates = 1
        
        if not self.noCandidates:
            self.check_subsumed_tables(tables_aligned_tuple_indxes)
        return self.candidateTablesFound, self.noCandidates
    