import pandas as pd
import numpy as np

# convert the <UserId, ItemId, prediction> format to horizontal: <UserId, Reco1, Reco2, ..., Reco5>
def long2shortRecos(df_long, k_max = 5):
    k_max = k_max
    D_recos = []
    pre = ""
    sim_items = []
    k = 1
    for row in df_long.itertuples():
        UserId = str(row[1])
        ItemId = str(row[2])
    
        # for a new user block" start except the very first one
        if (pre == ""):
            sim_items.append(UserId)
        if (pre != UserId and pre != ""): 
            D_recos.append(sim_items)
            sim_items = []
            sim_items.append(UserId) 
            k = 1
        if(k <= k_max):
            pre=UserId   
            sim_items.append(ItemId)
            k +=1
        else:
            pass
    # last item related items
    D_recos.append(sim_items)

    df_recos_h = pd.DataFrame(D_recos, dtype='str')
    return df_recos_h


# helper function 
def listOverlap(list1, list2):
    # This function computes the number of overlapped items in two lists
    return list(set(list1) & set(list2))

# compute precision@K
def precisionK(testDF = None, predictDF = None, k_max = 5, itemIdIgnoreCase = True):
    # Step 1. Create a dictionary from the test ground truth file
    D_test = {}
    for row in testDF.itertuples():
        userId = str(row[1])
        itemId = str(row[2])
        if itemIdIgnoreCase:
            itemId = itemId.upper()
        
        if userId not in D_test:
            D_test[userId] = []
        D_test[userId].append(itemId)
    total_user_num = len(D_test.keys())
    total_overlap_user_num = 0
    
    # Step 2. Go through the predictDF and compute precision accordingly
    precision_sum_list = [0 for i in range(k_max)]
    user_count_num_list = [0 for i in range(k_max)]
    user_with_k_rating_list = [0 for i in range(k_max)]
    for i in range(predictDF.shape[0]):
        userId = str(predictDF.ix[i,0])
        itemIdList = predictDF.ix[i,1:].tolist()
        # Process ignoreItemIdCase
        if itemIdIgnoreCase:
            itemIdList = [str(itemId).upper() for itemId in itemIdList]
        else:
            itemIdList = [str(itemId) for itemId in itemIdList]
        # Filter out empty recomendation
        itemIdList_2 = [itemId for itemId in itemIdList if (itemId != '' and itemId != '""' and itemId != 'NONE')]
        
        if userId in D_test:
            total_overlap_user_num += 1
            g_test = D_test[userId]
           
            for k in range(k_max):
                pg_test = listOverlap(g_test, itemIdList_2[:k+1])
                if len(pg_test)>0:
                    p_u = 1.0
                else:
                    p_u = 0.0
                precision_sum_list[k] = precision_sum_list[k] + p_u
                user_count_num_list[k] = user_count_num_list[k] + 1
                if len(itemIdList_2) >= (k+1):
                    user_with_k_rating_list[k] = user_with_k_rating_list[k] + 1

    # Step 3. Compute precision@k for different k values
    precision_avg_list = [0.0 for k in range(k_max)]
    for i in range(k_max):
        precision_avg_list[i] = float(precision_sum_list[i]) / user_count_num_list[i]
    # Fill the precision results as a pandas data frame and then return it.
    k_list = [1+i for i in range(k_max)]
    outputDict = {'k': k_list, \
    'precision': precision_avg_list, \
    'totalUserCount': total_overlap_user_num, \
    'totalUserCountConsidered': user_with_k_rating_list}
    outputDF = pd.DataFrame(outputDict)
    # Return value must be of a sequence of pandas.DataFrame
    return outputDF


def precision_mpi(testDF = None, predictDF = None, k_max = 5, itemIdIgnoreCase = True):
    # Step 1. Create a dictionary from the test ground truth file
    D_test = {}
    for row in testDF.itertuples():
        userId = str(row[1])
        itemId = str(row[2])
        if itemIdIgnoreCase:
            itemId = itemId.upper()
        
        if userId not in D_test:
            D_test[userId] = []
        D_test[userId].append(itemId)

    # Step 2. Go through the predictDF and compute precision accordingly
    precision_sum_list = [0 for i in range(k_max)]
    user_count_num_list = [0 for i in range(k_max)]  
    k = 0
    topn_itemId_list = [] # store most popular items
    for i in range(predictDF.shape[0]):
        if(k < k_max):
            itemId = predictDF.ix[i,0]      
            topn_itemId_list.append(itemId)
            # Process ignoreItemIdCase
            if itemIdIgnoreCase:
                topn_itemId_list = [str(itemId).upper() for itemId in topn_itemId_list]
            else:
                topn_itemId_list = [str(itemId) for itemId in topn_itemId_list]
        else:
            break
        k+=1      
    
        # Filter out empty recomendation
        topn_itemId_list = [itemId for itemId in topn_itemId_list if (itemId != '' and itemId != '""')]
        
    for userId in D_test:
        g_test = D_test[userId]
        for k in range(k_max):
          pg_test = listOverlap(g_test, topn_itemId_list[:k+1])
          if len(pg_test)>0:
              p_u = 1.0
          else:
              p_u = 0.0
          precision_sum_list[k] = precision_sum_list[k] + p_u
          user_count_num_list[k] = user_count_num_list[k] + 1

    # Step 3. Compute precision@k for different k values
    precision_avg_list = [0.0 for k in range(k_max)]
    for i in range(k_max):
        precision_avg_list[i] = float(precision_sum_list[i]) / user_count_num_list[i]
    # Fill the precision results as a pandas data frame and then return it.
    k_list = [1+i for i in range(k_max)]
    outputDict = {'k': k_list, 'precision': precision_avg_list, 'totalUserCount': user_count_num_list[0], 'totalUserCountConsidered': user_count_num_list}
    outputDF = pd.DataFrame(outputDict)
    # Return value must be of a sequence of pandas.DataFrame
    return outputDF
