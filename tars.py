

#inter time and intra time
import math
from itertools import *
import numpy as np
from __future__ import division
from collections import Counter
import operator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


new = pd.merge(singleuser_with_orderlist,orders_df,on =['order_id','user_id'], how= 'left')



def intra_inter_time(sorted_transactions_df,pattern,del_min,qmin) :
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
                        elif len(period) >= qmin:
                            periods_list.append(period)
                            period = []
                        else :
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
    if len(period) >= qmin:
        periods_list.append(period)

    return intra,inter,periods_list

def del_max(sorted_transactions_df,pat) :
    pats = [item[0] for item in pat.items()]
    inter_all = []
    max_score = -1
    best_n = 2
    for pat in pats :
        intra,inter,periods = intra_inter_time(sorted_transactions_df,pat,2,0)
        inter_max = max(inter)
        inter_all.append(inter_max)

    inter_all = np.array(inter_all).reshape(-1,1)
    #clustering

    range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10]
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(inter_all)
        s_avg = silhouette_score(inter_all, cluster_labels)
        if  s_avg > max_score :
            max_score = s_avg
            best_n = n_clusters

    clusterer = KMeans(n_clusters=best_n, random_state=10)
    cluster_labels = clusterer.fit_predict(inter_all)
    df = pd.DataFrame()
    df['pats'] = pats
    df['inter_max'] = inter_all
    df['del_cluster_labels'] = cluster_labels
    df2 = df.groupby(['del_cluster_labels']).apply(lambda x : np.median(x['inter_max'])).reset_index()
    df3 = pd.merge(df, df2, on='del_cluster_labels', how='left')
    df3 = df3.rename(columns={0: 'assigned_inter_max'})

    return df3

def q_min(sorted_transactions_df, df) :
    pats = df['pats'].tolist()
    del_assigned = df['assigned_inter_max']
    medians = []
    max_score = -1
    best_n = 2
    for y in range(len(pats)) :
        intra,inter,periods = intra_inter_time(sorted_transactions_df,pats[y],del_assigned[y],0)
        periods_lens = [len(p) for p in periods]
        if len(periods_lens) == 0 :
            medians.append(0)
        else :
            median_occ = np.median(periods_lens)
            medians.append(median_occ)

    medians = np.array(medians).reshape(-1, 1)

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(medians)
        s_avg = silhouette_score(medians, cluster_labels)
        if s_avg > max_score:
            max_score = s_avg
            best_n = n_clusters

    clusterer = KMeans(n_clusters=best_n, random_state=10)
    labels = clusterer.fit_predict(medians)
    df['medians'] = medians
    df['q_cluster_labels'] = labels
    df2 = df.groupby(['q_cluster_labels']).apply(lambda x: np.median(x['medians'])).reset_index()
    df3 = pd.merge(df, df2, on='q_cluster_labels', how='left')
    df3 = df3.rename(columns={0: 'assigned_q_min'})

    return df3




dd = del_max(final_df,patrns)

dp = q_min(final_df,dd)

intra,inter,periods = intra_inter_time(new,pattern=pat.items()[1][0],del_min= 70,qmin=0)

def p_min(new,pat):
    overall = []
    pats = [item[0] for item in pat.items()]
    for pat in pats :
        intra,inter,periods = intra_inter_time(new,pat,2,0)
        if len(inter)  != 0 :
            pmi = np.percentile(inter,80)
        else :
            pmi = 5
        intra, inter, periods = intra_inter_time(new, pat, pmi)
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
        if len(inter) != 0 :
            pmq = np.percentile(inter, 80)
        else :
            pmq = 5
        intra, inter, periods = intra_inter_time(df,i,pmq)
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

