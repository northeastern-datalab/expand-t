import numpy as np
import pandas as pd
import time
import glob
from preprocess import getTableDfs, preprocessForeignKeys
import sys
sys.path.append('../')
from utils import projectAtts, selectKeys
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)    
pd.options.mode.chained_assignment = None

class OriginatingTables:
    def __init__(self, benchmark, candidate_tables={}):
        self.benchmark = benchmark
        self.candidate_tables = candidate_tables
        self.candidate_table_dfs = {}
        self.source_table = pd.DataFrame()
        self.primary_key = None
        self.foreign_keys = []
        self.timesStats = {}        

    def initializeTableMatrices(self, tableDfs):
        ''' Create matrixes for each data lake table, 1 = value matches source table, 0 = doesn't match, -1 = value is non-null and doesn't match
        Input:
            tableDfs (dict): tableName and table Df
        Return dictionary of each candidate table with its matrix representation
        '''
        tableMatrices = {}
        for table, df in tableDfs.items():
            dfMatrix = {k: np.zeros((1, self.source_table.shape[1])).tolist() for k in self.source_table[self.primary_key].values} # dictionary: source keys: matrix
            for indx, qKey in enumerate(self.source_table[self.primary_key].values):
                if pd.isna(qKey): continue # skip if the source table's primary key value is Null
                i = self.source_table.index[self.source_table[self.primary_key]==qKey].tolist()[0]
                commonKey = self.primary_key
                commonKeys = [k for k in [self.primary_key]+self.foreign_keys if k in df.columns]
                if len(commonKeys) > 0: commonKey = commonKeys[0]
                qKeyVal = self.source_table.iloc[i, self.source_table.columns.tolist().index(commonKey)]
                numAddedCols = 0
                if qKeyVal in df[commonKey].values:
                    if df.loc[df[commonKey] == qKeyVal].shape[0] > 1: 
                        for _ in range(df.loc[df[commonKey] == qKeyVal].shape[0]-1):
                            dfMatrix[qKey] += np.zeros((1, self.source_table.shape[1])).tolist()
                            numAddedCols += 1
                            
                for qCol in self.source_table.columns:
                    sourceVal = self.source_table.loc[self.source_table[self.primary_key] == qKey, qCol].tolist()[0]
                    # get i, j index of matrix
                    if qCol not in df.columns: continue
                    j = list(self.source_table.columns).index(qCol)
                    
                    if qKeyVal in df[commonKey].values:
                        if df.loc[df[commonKey] == qKeyVal].shape[0] > 1: 
                            # there are multiple rows in DataFrame with same key
                            dfVals = df.loc[df[commonKey] == qKeyVal, qCol].values.tolist()
                        elif type(df.loc[df[commonKey] == qKeyVal, qCol]) == pd.DataFrame: dfVals = df.loc[df[commonKey] == qKeyVal, qCol].values.tolist()[0]
                        else: dfVals = df.loc[df[commonKey] == qKeyVal, qCol].tolist()
                        # check NaNs
                        for valIndx, val in enumerate(dfVals):
                            if pd.isna(val) and pd.isna(sourceVal): # both are NULL
                                dfMatrix[qKey][valIndx][j] = 1.0
                            if val == sourceVal:
                                dfMatrix[qKey][valIndx][j] = 1.0
                            if val != sourceVal and not pd.isna(val):
                                dfMatrix[qKey][valIndx][j] = -1.0                        
                                
                    elif self.primary_key not in df.columns or (self.primary_key in df.columns and qKey not in df[self.primary_key].values):
                        if sourceVal in set(df[qCol].values): 
                            # df does not contain primary key but overlaps property values
                            dfMatrix[qKey][0][j] = 1.0
                    if pd.isna(sourceVal):
                        dfMatrix[qKey][0][j] = 1.0
            tableMatrices[table] = dfMatrix
        return tableMatrices
            
    def combineTernaryMatrices(self, aTable, bTable):
        '''
        combine ternary matrices
        '''
        combinedMatrix = {}
        for key, aRows in aTable.items():
            combinedMatrix[key] =  []
            bRows = bTable[key] # aRows, bRows = list of lists
            toCombineRows = {} # list of Booleans, if exists False then don't combine rows
            for colIndx in range(len(aRows[0])):
                aVals, bVals = [row[colIndx] for row in aRows], [row[colIndx] for row in bRows]
                for aIndx, aVal in enumerate(aVals):
                    for bIndx, bVal in enumerate(bVals):
                        if (aIndx, bIndx) not in toCombineRows: toCombineRows[(aIndx, bIndx)] = []
                        if aVal != bVal and aVal != 0 and bVal != 0:
                            # if they are -1 / 1
                            toCombineRows[(aIndx, bIndx)].append(False)
                        else: toCombineRows[(aIndx, bIndx)].append(True)
            
            combinedAIndexes, combinedBIndexes = set(), set()
            for combIndx, toCombine in toCombineRows.items():
                # combine if all equal, max() if there exists a 0
                # DO NOT combine if 1 and -1 are both there
                if all(toCombine):
                    combRow = np.maximum(aRows[combIndx[0]], bRows[combIndx[1]]).astype(float).tolist()
                    combinedMatrix[key].append(combRow)
                    combinedAIndexes.add(combIndx[0])
                    combinedBIndexes.add(combIndx[1])
                else:
                    if combIndx[0] not in combinedAIndexes: combinedMatrix[key].append(aRows[combIndx[0]])
                    if combIndx[1] not in combinedBIndexes: combinedMatrix[key].append(bRows[combIndx[1]])
            
        return combinedMatrix
        

    def traverseGraph(self, tableMatrices, startTable):
        '''
        Traverse space of matrices to and combine pairs of matrices,
        end when the resulting matrix is all 1's or all matrices have been combined
        Return:
            list of tables whose matrix representations were combined, and percentage of 1s in resulting matrix
        '''
        startTable, startMatrix = startTable, tableMatrices[startTable]
        traversedTables, nextTable = [startTable], None
        # Using Normalized VSS as evaluateSimilarity()
        prevCorrect = mostCorrect = self.findPercentageCorrect_norm(startMatrix)
        
        testCount = 0
        exitEarly = 0
        while len(traversedTables) < len(tableMatrices) and mostCorrect < 1.0 and not exitEarly:
            startTime = time.time()
            prevCorrect = mostCorrect
            testCount += 1
            for table, matrix in tableMatrices.items():
                if table not in traversedTables:
                    intermediateMatrix = startMatrix
                    if len(traversedTables) > 1: 
                        for tTable in traversedTables:
                            intermediateMatrix = self.combineTernaryMatrices(intermediateMatrix, tableMatrices[tTable])                
                    combinedMatrix = self.combineTernaryMatrices(intermediateMatrix, matrix)
                    # Using Normalized VSS metric as evaluateSimilarity()
                    percentCorrectVals = self.findPercentageCorrect_norm(combinedMatrix)
        
                    
                    if percentCorrectVals > mostCorrect:
                        mostCorrect = percentCorrectVals
                        nextTable = table
            if mostCorrect == prevCorrect: exitEarly = 1 # iterated through all tables, and no improvement
            if not exitEarly:
                traversedTables.append(nextTable)
        return traversedTables, mostCorrect


    def findPercentageCorrect_norm(self, matrixDict):
        '''
        evaluate Similarity: find the percentage of 1's or correct values in the current matrix
        '''
        checkTuples = []
        for key, tuples in matrixDict.items():
            if len(tuples) == 1: 
                checkTuples += tuples
            else:
                mostCorrectTuple, mostCorrectPercent = [], -0.1
                for t in tuples:
                    correctPercent = len([val for val in t if val>0]) / len(t)
                    if correctPercent > mostCorrectPercent:
                        mostCorrectPercent = correctPercent
                        mostCorrectTuple = t
                checkTuples.append(mostCorrectTuple)
        checkMatrix = [item for sublist in checkTuples for item in sublist]
        
        percentCorrectVals = len([val for val in checkMatrix if val>0]) / len(checkMatrix)
        percentErrVals = len([val for val in checkMatrix if val<0]) / len(checkMatrix)
        
        if percentCorrectVals == 0.0 or percentErrVals > percentCorrectVals: return None
        return 0.5*(1 + percentCorrectVals - percentErrVals)

    def getDLMatrices(self, tableDfs):
        '''
        initialize candidate tables as matrices to align their values with the Source Table,
        then combine matrices
        '''
        startTime = time.time()
        
        tableMatrices = self.initializeTableMatrices(tableDfs) 
        matrixInitTime = time.time() - startTime
        # pick start node
        startTime = time.time()
        startTable, mostCorrect = {}, 0
        tableCorrectVals = {}
        removeTables = []
        for table, matrix in tableMatrices.items():
            # Using Normalized VSS metric as evaluateSimilarity()
            percentCorrectVals = self.findPercentageCorrect_norm(matrix)
            
            if not percentCorrectVals: 
                removeTables.append(table)
                continue
            tableCorrectVals[table] = percentCorrectVals
            if percentCorrectVals > mostCorrect: 
                startTable = table
                mostCorrect = percentCorrectVals
        for table in removeTables:
            tableDfs.pop(table)
            tableMatrices.pop(table)
        if mostCorrect == 0.0: return None, None, None, None, None, None
        
        tableMatrices = {k: v for k, v in sorted(tableMatrices.items(),key = lambda item : tableCorrectVals[item[0]], reverse=True)}
        startTime = time.time()
        traversedTables, correctVals = self.traverseGraph(tableMatrices, startTable)
        matTraverseTime = time.time() - startTime
        self.timesStats['matrix_initialization'] = [matrixInitTime]
        self.timesStats['matrix_traversal'] = [matTraverseTime]
        return tableDfs, tableMatrices, traversedTables, correctVals
    

    def getPreprocessedTables(self, datasets, sourceTableName):
        # Get preprocessed tables
        self.candidate_table_dfs = getTableDfs(self.benchmark, datasets, sourceTableName, self.candidate_tables)
        if self.candidate_table_dfs is None: return self.candidate_table_dfs
        
        for table, df in self.candidate_table_dfs.items():
            # check types
            for col in df.columns:
                if col in self.source_table:
                    try: df.loc[:, col] = df[col].astype(self.source_table[col].dtypes.name)
                    except: 
                        df = df.dropna()
                        try: df.loc[:, col] = df[col].astype(self.source_table[col].dtypes.name)
                        except: df = df.drop(col, axis=1) # if cannot convert to same type, delete  

    def find_originating_tables(self,sourceTableName='source'):
        FILEPATH = '/home/gfan/Datasets/%s/' % (self.benchmark) 
        # ===== REAL Data Lake ====
        datasets = glob.glob(FILEPATH+'datalake/*.csv')
        self.source_table = pd.read_csv(FILEPATH+"queries/"+sourceTableName)
        # ==== TPTR Datalake
        # Primary Key is first column in dataframe
        self.primary_key = self.source_table.columns.tolist()[0]
        self.foreign_keys = [colName for colName in self.source_table.columns.tolist() if 'key' in colName and colName != self.primary_key]
        # ==== T2D_GOLD Datalake
        if 't2d_gold' in self.benchmark:
            self.primary_key = self.source_table.columns.tolist()[0]
            # Get another primary key if the first column only has NaN's
            if len([val for val in self.source_table[self.primary_key].values if not pd.isna(val)]) == 0:
                for colIndx in range(1, self.source_table.shape[1]):
                    currCol = self.source_table.columns.tolist()[colIndx]
                    if len([val for val in self.source_table[currCol].values if not pd.isna(val)]) > 1:
                        self.primary_key = currCol
                        break
            self.foreign_keys = []

        self.getPreprocessedTables(datasets, sourceTableName)
        if not self.candidate_table_dfs: return None, None
        
        projectedTableDfs = projectAtts(self.candidate_table_dfs, self.source_table)
        finalTableDfs = selectKeys(projectedTableDfs, self.source_table, self.primary_key, self.foreign_keys)
        finalTableDfs = preprocessForeignKeys(finalTableDfs, self.primary_key, self.foreign_keys, self.source_table)
        if not finalTableDfs: return None, None
        finalTableDfs, tableMatrices, traversedTables, correctVals = self.getDLMatrices(finalTableDfs)
        if tableMatrices == None and traversedTables == None and correctVals == None: self.timesStats = None
        
        # return list of traversed tables for Source as originating tables
        return traversedTables, self.timesStats
        