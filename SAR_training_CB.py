import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import datetime
import math


def SAR_Train_CB(df_jaccard = None, df_catalog = None, jaccard_idx = 5, featureValueSep = ";", k_max = 5, c_c = False):
    # 1. Compute normalized catalog feature weights
    df_feature_scores = feature_ranker(df_jaccard, df_catalog, jaccard_idx, featureValueSep)
    
    # 2. add ItemType to df_catalog - W(arm) if in the jaccard table, else C(old)
    warm_item_list = pd.concat([df_jaccard.ItemId1, df_jaccard.ItemId2],axis = 0).unique()
    df_catalog['ItemType'] = "C"
    for row in df_catalog.itertuples():
        iid = row[1]
        if(iid in warm_item_list):
            df_catalog.set_value(row[0],'ItemType', "W")
    print("DEBUG:\nFreq of ItemType:\n",df_catalog.ItemType.value_counts())

    # 3. Comppute content feature similarity among Cold and Warm/Cold items
    itemIdIgnoreCase = True
    df_feature_sim = feature_sim(df_catalog, df_feature_scores, k_max , c_c, itemIdIgnoreCase, featureValueSep)
    
    return df_feature_sim

    
# compute feature score on warm item similarities
def feature_ranker(df_jaccard = None, df_catalog = None, jaccard_idx = 5, featureValueSep = ";"):

    # Internal Parameter settings
    PearsonCorr = {}

    #Sometimes, the item ID column can be treated as of mixed type, this is safe measure
    df_jaccard[['ItemId1', 'ItemId2']] = df_jaccard[['ItemId1', 'ItemId2']].astype(str)
    
    #Remove unneeded columns
    df_jaccard =  df_jaccard[['ItemId1', 'ItemId2', 'jaccard']]
    df_catalog = df_catalog.drop(df_catalog.columns[[1]], axis=1)
    
    #Just in case we forget to convert everything into String in experiment
    for column in df_catalog:
        df_catalog[column] = df_catalog[column].astype(str)

    #Obtain the actual feature names
    featureNames = list(df_catalog.columns.values)[1:len(df_catalog)]
    #print("DEBUG: Features:", featureNames)
    #pd.DataFrame.copy makes a copy of the DF, instead of copying reference	
    df_catalog2 = df_catalog.copy(deep=True)

    #Need to make the key column names match when we do merge later	
    df_catalog.rename(columns={'ItemId': 'ItemId1'}, inplace=True)
    df_catalog2.rename(columns={'ItemId': 'ItemId2'}, inplace=True)

    #Step 1. Join Jaccard and feature DF to get Jaccard and Features for both items in each item pair into one DF
    DF_merged = pd.merge(pd.merge(df_jaccard,df_catalog,on='ItemId1'),df_catalog2,on='ItemId2')
 
    #Features should start from the 4th column (after ItemId1, ItemId2, and Jaccard)
    feature_idx_list = range(3, DF_merged.shape[1])

    #There are two sets of features, one from each item in a pair, so feature length is half of all features
    feature_len = int(len(feature_idx_list)/2)
    feature_idx_list = range(3, 3+feature_len)

    #Step 2. Calculate similarity between each feature pair	
    for fi in feature_idx_list: #Feature index
        sim_fi = []

        # If feature can be a list of values, the following evaluates feature similarity between two feature sets A & B via
        # len(intersect(A, B))/min(len(A), len(B))
        if (any(DF_merged.iloc[:, fi].str.contains(featureValueSep)) |
                any(DF_merged.iloc[:, fi+feature_len].str.contains(featureValueSep))):
            list1 = list(DF_merged.iloc[:, fi].str.split(featureValueSep))
            list2 = list(DF_merged.iloc[:, fi+feature_len].str.split(featureValueSep))

            for ei in range(len(list1)):
                # Remove non-informational feature values
                while 'N/A' in list1[ei]: list1[ei].remove('N/A')
                while 'N/A' in list2[ei]: list2[ei].remove('N/A')
                # Handle missing and empty feature comparison here
                if (min(len(list1[ei]), len(list2[ei])) == 0 | int('nan' in list1[ei]) | 
                    int('nan' in list2[ei]) | int('None' in list1[ei]) | int('None' in list2[ei])):
                    sim_fi.append(0)
                else:
                    sim_ele = len(set(list1[ei]).intersection(set(list2[ei])))/ float(min(len(list1[ei]), len(list2[ei])))

                    #This is for checking feature sets comparison
                    #if sim_ele > 0:
                    #    print(list1[ei])
                    #    print(list2[ei])
                    #    print(sim_ele)
                    sim_fi.append(sim_ele)
        else: # Otherwise, evaluate feature similarity by direct matching
            sim_fi = DF_merged[[fi]].values == DF_merged[[fi+feature_len]].values
            
            # Handle missing and empty feature comparison here
            sim_fi[(DF_merged[[fi]].values == 'None').astype(int) | (DF_merged.ix[[fi+feature_len]].values == 'None').astype(int)] = 0
            sim_fi[(DF_merged[[fi]].values == 'N/A').astype(int) | (DF_merged.ix[[fi+feature_len]].values == 'N/A').astype(int)] = 0
            sim_fi[(DF_merged[[fi]].values == '').astype(int) | (DF_merged.ix[[fi+feature_len]].values == '').astype(int)] = 0
            sim_fi[(DF_merged[[fi]].values == 'nan').astype(int) | (DF_merged.ix[[fi+feature_len]].values == 'nan').astype(int)] = 0
            sim_fi[(DF_merged[[fi]].values == 'NaN').astype(int) | (DF_merged.ix[[fi+feature_len]].values == 'NaN').astype(int)] = 0
            
            sim_fi = sim_fi.astype(int)

        try: #Calculate Pearson Correlation using the Numpy function
            PearsonCorr[fi] = np.corrcoef(np.ravel(sim_fi), np.ravel(DF_merged[['jaccard']]))[0, 1]
        except:
            PearsonCorr[fi] = 0
        if math.isnan(PearsonCorr[fi]):
            PearsonCorr[fi] = 0

    print("DEBUG: Raw weights:", PearsonCorr)
    
    # Step 3. Organize the output as a pandas data frame
    key_sorted = sorted(PearsonCorr.keys())
    for i in range(len(key_sorted)):
         PearsonCorr[key_sorted[i]] = max(0,PearsonCorr[key_sorted[i]])

    D_output = {(featureNames[i]):[float(PearsonCorr[key_sorted[i]])/sum(PearsonCorr.values())] for i in range(len(key_sorted))}
    DF_output = pd.DataFrame(D_output)
    print("DEBUG: Norm weights:\n", DF_output)

    return DF_output


#This function replaces non-informational features with empty string.
#eString is a feature column, where each feature in the column is a string
def cleanMissingValueFromStr(eString):
    eString[eString.values == 'N/A'] = ''
    eString[eString.values == 'nan'] = ''
    eString[eString.values == 'NaN'] = ''
    eString[eString.values == 'None'] = ''
    return eString

#This function cleans missing values from feature value lists. eList is a feature column.
#Each feature in the column can have a list of feature values
def cleanMissingValueFromList(eList):
    for iList in eList:  # Each iList itself is a list, need to iterate through it to remove all matching features
        # Missing feature can take different forms, nan, None, or '', and possibly others ...
        #print(iList)
        while 'N/A' in iList: iList.remove('N/A')
        while 'nan' in iList: iList.remove('nan')
        while 'Nan' in iList: iList.remove('NaN')
        while 'None' in iList: iList.remove('None')
        while '' in iList: iList.remove('')
    return eList

def feature_sim(cat=None, featureW=None, k_max = 5, c_c = False, itemIdIgnoreCase = True, featureValueSep = ";"):
    # Remove columns not needed to reduce memory footprint - remove ItemName & Category columns

    cat.drop(cat.columns[[1]], axis=1, inplace=True)

    if itemIdIgnoreCase:
        cat[cat.columns[0]] = cat[cat.columns[0]].str.upper()

    # get number of features, assume Catalog is in dense format, no Description column
    numFeatures = len(cat.ix[0, :]) - 2
    if (numFeatures < 1):
        outputLS = []
        return pd.DataFrame(outputLS)

    # Indices of feature columns
    feature_start_idx = 1
    feature_idx_list = range(feature_start_idx, feature_start_idx + numFeatures)
    cat[cat.columns[feature_idx_list]] = cat[cat.columns[feature_idx_list]].astype(str)

    coldItems = cat[(cat.ItemType == 'C')]

    if (c_c):  # If we want to also do c_c recommendation
        warmItems = cat
    else:  # If we only want to do c_w recommendation
        warmItems = cat[cat.ItemType == 'W']

    #print("DEBUG: Number of Warm Items: ", warmItems.shape)
    #print("DEBUG: Number of Cold Items: ", coldItems.shape)
    print("DEBUG: Number of Features: ", numFeatures)
    print("DEBUG: Feature Weight List: ", featureW)

    # Remove last column for item type (cold or warm)
    coldItems.drop(coldItems.columns[[len(coldItems.columns) - 1]], axis=1, inplace=True)
    warmItems.drop(warmItems.columns[[len(warmItems.columns) - 1]], axis=1, inplace=True)

    # Obtain the actual feature names
    featureNames = list(coldItems.columns.values)[1:(1 + numFeatures)]

    # Initialize output data frame
    outCols = ['ItemId', 'Sim', 'ItemId_c']
    cwTopPairs = pd.DataFrame(data=np.zeros((0, len(outCols))), columns=outCols)

    # This indicates whether a feature can be a list or not
    featureListInd = np.zeros(numFeatures)

    for fi in feature_idx_list:  # Feature index
        if (any(warmItems.iloc[:, fi].astype(str).str.contains(featureValueSep)) |
                any(coldItems.iloc[:, fi].astype(str).str.contains(featureValueSep))):
            featureListInd[fi - feature_start_idx] = 1
            warmItems[featureNames[fi - feature_start_idx]] = list(warmItems.iloc[:, fi].str.split(featureValueSep))
            coldItems[featureNames[fi - feature_start_idx]] = list(coldItems.iloc[:, fi].str.split(featureValueSep))

            # Remove non-informational/missing feature values
            #print(warmItems[featureNames[fi - feature_start_idx]])
            cleanMissingValueFromList(warmItems[featureNames[fi - feature_start_idx]])
            cleanMissingValueFromList(coldItems[featureNames[fi - feature_start_idx]])
        else:
            cleanMissingValueFromStr(warmItems[featureNames[fi - feature_start_idx]])
            cleanMissingValueFromStr(coldItems[featureNames[fi - feature_start_idx]])


    if not any(featureListInd):  #If none of the feature columns is a list, , evaluate feature similarity by direct matching
        for ci in coldItems.itertuples():
            cwPairs = pd.DataFrame.copy(warmItems)
            # Get weighted similarity for each feature
            cwPairs['Sim'] = 0
            for fi in feature_idx_list:  # Feature index
                if len(ci[fi + 1]) > 0:
                    # Sum up weighted feature similarity for all features for each cold-warm item pair
                    cwPairs[['Sim']] = cwPairs[['Sim']].values + (cwPairs[[fi]].values == ci[fi + 1]).astype(int) * featureW[[fi - 1]].values

            cwPairs = cwPairs[cwPairs['Sim'] > 0]
            if cwPairs.shape[0] > 0:
                cwPairs['ItemId_c'] = ci[1]

                # Obtain the top k_max + 1 warm items to pair with the cold item per similarity score
                # In case c_c = True, we may have the same cold item in top k_max + 1, need to remove it
                # so we start with k_max + 1 in order to get top k_max items
                cwPairs = cwPairs.sort_values(by=['Sim'], ascending=False)
                cwTopK = cwPairs.head(k_max + 1)

                cwTopK = cwTopK[cwTopK.ItemId != ci[1]]
                cwTopK = cwTopK.head(k_max)

                if cwTopK.shape[0] > 0:
                    # Keep only the columns needed for output
                    cwTopPairs = cwTopPairs.append(cwTopK[['ItemId', 'Sim', 'ItemId_c']])
    else:
        for ci in coldItems.itertuples():
            #startTime = datetime.now()
            # Be careful: tuple ci will return index as an extra first column

            cwPairs = pd.DataFrame.copy(warmItems)
            cwPairs['Sim'] = 0

            # Get weighted similarity for each feature
            for fi in feature_idx_list:  # Feature index
                #If either cold item's feature contains a list or any warm items' feature contains lists, we
                #evaluate feature similarity between two feature sets A & B via: len(intersect(A, B))/min(len(A), len(B))
                if featureListInd[fi - feature_start_idx]:
                    wList = cwPairs[featureNames[fi - feature_start_idx]]
                    cList = ci[fi + 1]

                    cwPairs['S_' + str(fi)] = 0
                    if len(cList) > 0:
                        sim_fi = [len(set(cList).intersection(set(subList))) /
                                                   float(min(len(cList), len(subList))) if len(subList) > 0 else 0 for
                                                   subList in wList] * featureW[[fi - 1]].values
                        cwPairs['S_' + str(fi)] = list(sim_fi[0])
                else:
                    if len(ci[fi + 1]) == 0:
                        cwPairs['S_' + str(fi)] = 0
                    else:
                        cwPairs['S_' + str(fi)] = (cwPairs[[fi]].values == ci[fi + 1]).astype(int) * featureW[[fi - 1]].values

                # Sum up weighted feature similarity for all features for each cold-warm item pair
                cwPairs[['Sim']] = cwPairs[['Sim']].values + cwPairs[['S_' + str(fi)]].values

            cwPairs['ItemId_c'] = ci[1]

            # Obtain the top k_max + 1 warm items to pair with the cold item per similarity score
            # In case c_c = True, we may have the same cold item in top k_max + 1, need to remove it
            # so we start with k_max + 1 in order to get top k_max items
            cwPairs = cwPairs.sort_values(by=['Sim'], ascending=False)
            cwTopK = cwPairs.head(k_max + 1)

            cwTopK = cwTopK[cwTopK.ItemId != ci[1]]
            cwTopK = cwTopK.head(k_max)

            #Remove items with zero similarity
            cwTopK = cwTopK[cwTopK.Sim > 0]

            if cwTopK.shape[0] > 0:
                # Keep only the columns needed for output
                cwTopPairs = cwTopPairs.append(cwTopK[['ItemId', 'Sim', 'ItemId_c']])

    outputDF = cwTopPairs
    outputDF.columns = ['ItemId1', 'SimScore', 'ItemId2']

    # DF that holds the output in blocks: <ColdItemId, WarmItemId, SimScore>
    outputDF = outputDF.reindex(columns=['ItemId1', 'ItemId2', 'SimScore'])
    # Return value must be of a sequence of pandas.DataFrame
    return outputDF


