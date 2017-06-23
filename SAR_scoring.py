import pandas as pd
import numpy as np

def SAR_Score(simDF=None, usageDF=None, topN=5, actuals=False, fill=True, history=False, sim_idx=6):
    
    outputLS = []
    
    numItems = pd.concat([simDF.ItemId1, simDF.ItemId2],axis = 0).unique().shape[0]
    print("DEBUG: There are %s unique items.\n"%(numItems))
    
    # initialize a dictionary/hash to map item ids to consecutive ints starting at 0
    ilist={}
    iIndex=0
    # initialize a 2-D array to store item-item co-occurence counts and a 1-D to store Item occurrences
    simMatrix = np.zeros(numItems*numItems).reshape(numItems,numItems)
    totalOccur = np.zeros(numItems) 
    # iterate over the simDF and populate non-zero values in 
    for row in simDF.itertuples():
        Item1 = str(row[1])
        Item2 = str(row[2])     
        #if(iIndex == 0):
        #    print(row[1],row[2],row[6])
        try:
            # check if iid was added to ilist dictionary
            ilist[Item1]
        except KeyError:	
            # if iid not in ilist, add iid as key and consecutive int iIndex as value
            ilist[Item1]=iIndex  
            iIndex+=1
        try:
            # check if iid was added to ilist dictionary
            ilist[Item2]
        except KeyError:	
            # if iid not in ilist, add iid as key and consecutive int iIndex as value
            ilist[Item2]=iIndex  
            iIndex+=1
        
        totalOccur[ilist[Item1]] = row[4]
        totalOccur[ilist[Item2]] = row[5]
        
        simMatrix[ilist[Item1]][ilist[Item2]] = simMatrix[ilist[Item2]][ilist[Item1]] = row[sim_idx]
    
    # get the opposite of ilist, where key = mapped itemid, value = original itemid        
    rItem = {}
    for k,v in ilist.items():
        iid = k
        mapIID = v
        rItem[mapIID] = iid
        
    # store topN most popular items                     
    topNItems=np.argsort(totalOccur)[::-1][:topN]
    print("DEBUG: Most Popular Items:")
    for j in topNItems:
        print("DEBUG: %s\t%d"%(rItem[j],totalOccur[j]))

    sparsity = float(len(simMatrix.nonzero()[0]))
    sparsity /= (simMatrix.shape[0] * simMatrix.shape[1])
    sparsity *= 100
    print ("\nDEBUG: Sparsity of Item Similarity Matrix%: ",sparsity)
 

    # now, process the seed file to score, one user block at a time
    numUsers = usageDF.UserId.unique().shape[0]
    print("\nDEBUG: There are %s users to score."%(numUsers))
        
    # a list to store all items rated and an array to store values
    itemSet = []
    itemValues = np.zeros(numItems)
    pre_uid=""
    # reads from the input three-column file one line at a time, sorted by userId blocks
    for row in usageDF.itertuples():
        uid = str(row[1])
        iid = str(row[2])
        value = float(row[3])
        if(value <= 0):
            continue
        # for a new "user block" start except the very first one
        if (pre_uid != uid and pre_uid != ""): 
            if(len(itemSet) > 0):
                rec = (itemSet,itemValues)
                scoretop(simMatrix,ilist,rItem,rec,pre_uid,topNItems,outputLS,actuals,fill,history)
            elif(fill): # backfill 0 reco users
                n = 0
                for j in topNItems:
                    if(n < topN):
                        outputLS.append((pre_uid,rItem[j],0.001,"T"))
                        n +=1
                    else:
                        break   
            
            itemSet = []
            itemValues = np.zeros(numItems)
           
        pre_uid=uid
        if(iid in ilist):
            itemSet.append(iid) # append raw iid to list itemSet
            itemValues[ilist[iid]] = value
  
    if(len(itemSet) > 0):
        rec = (itemSet,itemValues)
        scoretop(simMatrix,ilist,rItem,rec,pre_uid,topNItems,outputLS,actuals,fill,history)
    elif(fill): # backfill 0 reco users
        n = 0
        for j in topNItems:
            if(n < topN):
                outputLS.append((pre_uid,rItem[j],0.01,"T"))
                n +=1       
            else:
                break
    
    outputDF = pd.DataFrame(outputLS)
    outputDF.columns = ['UserId', 'ItemId', 'Score', 'Flag']
  
    return outputDF
        

# SAR Scoring user for affinity to items no reasons
def scoretop(itemCoocs,ilist,rItem,rec,uid,topNItems,outputLS,actuals,fill,history):
    #print rec
    (ratedItems,ratedValues) = rec
    # if ratedItems is a dict, sort by value desc
    #ratedItemsS = sorted(ratedItems, key=ratedItems.get, reverse=True)

    # print the historical for this user first
    recosItemsHash = {}
    ratedItemsHash = {}
    for iid in ratedItems:
        ratedItemsHash[iid] = 1
        #print "%s\t%s\t%.1f\tH" % (uid,iid,ratedValues[ilist[iid]])
        if history:
            outputLS.append((uid,iid,ratedValues[ilist[iid]],"H"))
  
    # multiply the co-occur matrix with the user vector ratedValues
    numItems = len(ilist)
    simValues = np.zeros(numItems)
    #simValues = np.dot(itemCoocs,ratedValues) # this is an expensive operation, use below instead
    nonzero = np.nonzero(ratedValues)
    for i in nonzero[0]:
        temp = np.multiply(ratedValues[i],itemCoocs[ : ,i])
        simValues += temp
    
    topN = len(topNItems)
    top = topN + len(ratedItemsHash)
    indv = []
    #indv=np.argsort(simValues)
    indv=np.argsort(simValues)[::-1][:top]
    cnt=0
    for i in indv:
        if(cnt==0):
            max_scr = simValues[i] + 0.1
        iid = rItem[i]
        if(not actuals and iid in ratedItemsHash):  
            pass
        else: # 
            if(cnt < topN):  # restrict to topN highest scores
                scr = simValues[i]
                if(scr > 0):
                    scr = scr/max_scr
                    if(actuals and iid in ratedItemsHash):
                        #print "%s\t%s\t%.4f\tA" % (uid,iid,scr)
                        outputLS.append((uid,iid,scr,"A"))
                    else:
                        #print "%s\t%s\t%.4f\tR" % (uid,iid,scr)
                        outputLS.append((uid,iid,scr,"R"))
                        recosItemsHash[iid] = 1
                elif(fill): # backfill with topN items across population
                    for j in topNItems:
                        if(not (rItem[j] in ratedItemsHash) and not (rItem[j] in recosItemsHash)):
                            #print "%s\t%s\t%.4f\tT" % (uid,rItem[j],0.01)
                            outputLS.append((uid,rItem[j],0.01,"T"))
                            recosItemsHash[rItem[j]] = 1
                            break
                cnt+=1
            else:
                break