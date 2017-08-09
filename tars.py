

#inter time and intra time
import math
from itertools import *

new = pd.merge(singleuser_with_orderlist,orders_df,on =['order_id','user_id'], how= 'left')


def inter_time(sorted_transactions_df,pattern) :
    time = 0
    inter = []
    first = 0
    x = int(pattern.split(',')[0])
    for index,row in sorted_transactions_df.iterrows() :
        if x in row[0]  :
            if first == 0 :
                first = 1
                time = 0
            else :
                time = time + row['days_since_prior_order']
                inter.append(time)
                time = 0
        else :
            time = time + row['days_since_prior_order']

    return inter



inter = inter_time(new,pattern=pat.items()[0][0])


def intra_time(sorted_transactions_df,pattern) :
    time = 0
    intra = []
    first = 0
    x = int(pattern.split(',')[0])
    y = int(pattern.split(',')[1])
    i = 0
    for index,row in sorted_transactions_df.iterrows() :
        if x in row[0] :
            for index2,row2 in islice(sorted_transactions_df.iterrows(),i+1, None) :
                #print index
                #print index2
                if y in row2[0] :
                    time = time + row2['days_since_prior_order']
                    intra.append(time)
                    time = 0
                    #print row[0]
                    #print row2[0]
                    #print index
                    break
                else :
                    time = time + row2['days_since_prior_order']
        i = i+ 1
    return intra


intr = intra_time(new,pattern=pat.items()[0][0])



"""
Junk code



for index,row  in new.iterrows() :

    print row['days_since_prior_order'],row[0] 



for index, row in islice(new.iterrows(), 2, None) :
    print row['days_since_prior_order'], row[0]


for index,row  in new.iterrows() :

   if 19051 in row[0] :
        print index
 

    print row['days_since_prior_order'],row[0]


"""