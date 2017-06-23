import pandas as pd
import sqlite3

def SAR_Train(df_usage = None, df_catalog = None, supTh = 3, basket = "all"):
    print("DEBUG: Support Threshold:", supTh)
    # 0. Create your connection and populate Usage and Catalog tables.
    con = sqlite3.connect(':memory:')

    cur = con.cursor()

    cur.execute("CREATE TABLE Usage (UserId TEXT, ItemId Text, Date DATE);")
    cur.executemany("INSERT INTO Usage (UserId, ItemId, Date) VALUES(?,?,?)", 
                    list(df_usage[['UserId', 'ItemId', 'Date']].to_records(index=False)))

    cur.execute("CREATE TABLE Catalog (ItemId Text, ItemName Text);")
    cur.executemany("INSERT INTO Catalog (ItemId, ItemName) VALUES(?,?)", 
                    list(df_catalog[['ItemId', 'ItemName']].to_records(index=False)))
    print("DEBUG: 0. Usage and Catalog data loading DONE")

    # 1. Filter Usage to contain Catalog Items only and create basket type - "all": all user history is one basket
    if(basket == "all"):
        cur.execute("CREATE TABLE basket AS SELECT DISTINCT Usage.UserId AS UserId, Usage.ItemId AS ItemId \
                                            FROM Usage,Catalog \
                                            WHERE Usage.ItemId = Catalog.ItemId") 
    else:
        cur.execute("CREATE TABLE basket AS SELECT DISTINCT (Usage.UserId || Usage.Date) AS UserId, Usage.ItemId AS ItemId \
                                            FROM Usage,Catalog \
                                            WHERE Usage.ItemId = Catalog.ItemId")
    print("DEBUG: 1. Filtering and basket creation DONE. Basket type:", basket)

    # 2. compute item-item co-occurrences thresholded
    cur.execute("CREATE TABLE cooc AS select a.ItemId as ItemId1, b.ItemId as ItemId2, count(*) as cooc \
                from basket as a, basket as b \
                where a.UserId = b.UserId and a.ItemId > b.ItemId \
                group by ItemId1, ItemId2 \
                having cooc >=" + str(supTh))
    print("DEBUG: 2. Computing item-item co-occurrences DONE")

    # 3. compute item occurrences
    cur.execute("CREATE TABLE item_counts AS select ItemId, count(*) as occur from basket group by ItemId order by occur desc;")
    print("DEBUG: 3. Computing item occurrences DONE")

    # 4. compute item-item jaccard similarity
    cur.execute("CREATE TABLE jaccard AS select t1.ItemId1, t1.ItemId2, t1.cooc, \
               t2.occur as occur1, t3.occur as occur2, \
               (1.0*t1.cooc/(t2.occur+t3.occur-t1.cooc)) as jaccard \
                from item_counts t2 inner join cooc t1 on t1.ItemId1 = t2.ItemId inner join item_counts t3 on t1.ItemId2 = t3.ItemId \
                order by t1.ItemId1, jaccard desc;")
    print("DEBUG: 4. Computing item-item similarity DONE")

    # 5. convert item similarities to df
    df_jaccard = pd.read_sql_query("SELECT * from jaccard", con)
    df_mpi = pd.read_sql_query("SELECT ItemId as ItemId, occur \
                                        from item_counts \
                                        order by occur desc \
                                        limit 10", con)
    print("DEBUG: 5. Converting item-item similarities to df DONE")

    # 6. clean up teh sql tables and close the connection
    cur.execute("DROP TABLE Usage")
    cur.execute("DROP TABLE Catalog")
    cur.execute("DROP TABLE basket")
    cur.execute("DROP TABLE cooc")
    cur.execute("DROP TABLE item_counts")
    cur.execute("DROP TABLE jaccard")
    #con.commit()
    con.close()
    
    return df_jaccard, df_mpi
