

#inter time and intra time
import math
from itertools import *
import numpy as np
from __future__ import division
from collections import Counter
new = pd.merge(singleuser_with_orderlist,orders_df,on =['order_id','user_id'], how= 'left')

def inter_time(sorted_transactions_df, pattern, del_min):
    time = 0
    inter = []
    first_x = 0
    first_y = 0
    periods_list = []
    period = []
    next_y = 0
    x = int(pattern.split(',')[0])
    y = int(pattern.split(',')[1])
    i = 0
    for index, row in sorted_transactions_df.iterrows():

        if y in row[0]:
            first_y = 1
        if x in row[0]:
            for index2, row2 in islice(sorted_transactions_df.iterrows(), i + 1, None):
                if y in row2[0]:
                    next_y = 1

            if first_x == 0 or first_y == 0:
                first_x = 1
                time = 0
                last = row['order_number']
            else:
                time = time + row['days_since_prior_order']
                inter.append(time)
                if time < del_min:
                    period.append(last)
                elif len(period) != 0:
                    periods_list.append(period)
                    period = []

                last = row['order_number']
                time = 0
        else:
            time = time + row['days_since_prior_order']

        i = i + 1

    if len(period) != 0 :
        periods_list.append(period)


    return inter,periods_list



inter,periods = inter_time(new,pattern=pat.items()[1][0],del_min=45)




intr = intra_time(new,pattern=pat.items()[1][0])


def intra_time(sorted_transactions_df, pattern):
    time = 0
    intra = []
    first = 0
    x = int(pattern.split(',')[0])
    y = int(pattern.split(',')[1])
    i = 0
    for index, row in sorted_transactions_df.iterrows():
        if x in row[0]:
            for index2, row2 in islice(sorted_transactions_df.iterrows(), i + 1, None):
                if x in row2[0] and first == 0:
                    break
                if y in row2[0]:
                    time = time + row2['days_since_prior_order']
                    first = 1
                    intra.append(time)
                    time = 0
                    b = 1
                    break
                else:
                    time = time + row2['days_since_prior_order']

        i = i + 1
    return intra


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



def intra_inter_time(sorted_transactions_df,pattern,del_min) :
    time_intra = 0
    time_inter = 0
    last = 0
    intra = []
    inter = []
    period = []
    periods_list = []
    x = int(pattern.split(',')[0])
    y = int(pattern.split(',')[1])
    i = 0
    for index,row in sorted_transactions_df.iterrows() :
        if x in row[0] :
            if i != 0 :
                time_inter = time_inter + row['days_since_prior_order']
            last = row['order_number']
            for index2,row2 in islice(sorted_transactions_df.iterrows(),i+1, None) :
                if x in row2[0] and y not in row2[0]:
                    #time_inter = 0
                    break
                if y in row2[0] :
                    time_intra = time_intra + row2['days_since_prior_order']
                    intra.append(time_intra)
                    if len(intra) > 1 :
                        inter.append(time_inter)

                        if time_inter < del_min and last != 0:
                            period.append(last)
                        elif len(period) != 0:
                            periods_list.append(period)
                            period = []
                    last = row['order_number']
                    time_inter = 0
                    time_intra = 0
                    break
                else :
                    time_intra = time_intra + row2['days_since_prior_order']
        else:
            if i != 0 :
                time_inter = time_inter + row['days_since_prior_order']
        i = i + 1
    if len(period) != 0:
        periods_list.append(period)

    return intra,inter,periods_list


intra,inter,periods = intra_inter_time(new,pattern=pat.items()[1][0],del_min= 70)

def p_min(new,pat):
    overall = []
    pats = [item[0] for item in pat.items()]
    for pat in pats :
        intra,inter,periods = intra_inter_time(new,pat,90)
        #print intra
        #print inter
        #print periods
        overall.append(len(periods))

    if len(overall) > 0  :
        return np.percentile(overall,20)
    else :
        return 0


pm = p_min(new,pat)



def tbp_predictor(df,pat,d_min,pm) :
    Q = 0
    pats = [item[0] for item in pat.items()]
    tot_items = [m.split(',') for m in pats]
    tot_items = [item for sublist in tot_items for item in sublist]
    predictors = Counter(tot_items)
    for i in pats :
        intra, inter, periods = intra_inter_time(df,i,d_min)
        if len(periods)>= pm and len(periods)!=0:
            p= len(periods[len(periods)-1])
            q = np.median([len(it) for it in periods])
            if p ==q :
                Q = p
            else :
                Q = (p-q)/p

        kk = i.split(',')
        predictors[kk[0]] = predictors[kk[0]] + Q
        predictors[kk[1]] = predictors[kk[1]] + Q

    return dict(predictors)


kk = tbp_predictor(pat)


def final_product_list(sorted_transactions_df, items_dict) :
    sorted_items = sorted(items_dict.items(), key=operator.itemgetter(1),reverse = True)
    order_lengths = [len(it) for it in sorted_transactions_df[0]]
    median_size = int(np.mean(order_lengths))
    if median_size < len(sorted_items) :
        final_items = [int(sorted_items[i][0]) for i in range(median_size)]
    else :
        final_items = [int(item[0]) for item in sorted_items]

    return final_items




final_items = final_product_list(new,kk)


