import itertools

import os
import time
os.chdir('/root/mb/market_basket_data')
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score, confusion_matrix
import numpy as np # linear algebra
import pandas as pd

order_products_train_df = pd.read_csv("order_products__train.csv")
order_products_prior_df = pd.read_csv("order_products__prior.csv")
orders_df = pd.read_csv("orders.csv")
products_df = pd.read_csv("products.csv")
aisles_df = pd.read_csv("aisles.csv")
departments_df = pd.read_csv("departments.csv")

class FPNode(object):
    """
    A node in the FP tree.
    """
    def __init__(self, value, count, parent):
        """
        Create the node.
        """
        self.value = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.value == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.value == value:
                return node

        return None

    def add_child(self, value):
        """
        Add a node as a child node.
        """
        child = FPNode(value, 1, self)
        self.children.append(child)
        return child




def plist(orders) :
    pf_list = {}
    for index,order in enumerate(orders) :
        for product in order :
            if product in pf_list.keys() :
                pf_list[product]['freq'] += 1
                new_per = (index+1) - pf_list[product]['ts']
                if new_per > pf_list[product]['per'] :
                    pf_list[product]['per'] = new_per
                pf_list[product]['ts'] = index + 1

            else  :
                d = {}
                d['freq'] = 1
                d['per'] = index + 1
                d['ts'] = index + 1
                pf_list[product] = d
    for key in pf_list.keys():
        if len(orders) - pf_list[key]['ts'] > pf_list[key]['per'] :
            pf_list[key]['per'] = len(orders) - pf_list[key]['ts']

    #print pf_list
    return pf_list


prior_with_userids = pd.merge(order_products_prior_df,orders_df,on = 'order_id', how = 'left')

userids_with_orderlist = prior_with_userids.groupby(['user_id','order_id'])['product_id','order_number'].apply(lambda x: x['product_id'].tolist()).reset_index()

single_user_df = userids_with_orderlist[userids_with_orderlist['user_id']==1]

single_user_df = pd.merge(single_user_df,orders_df.iloc[:,[0,3]],on = 'order_id' , how = 'left')

single_user_df = single_user_df.sort_values(by ='order_number')

plist(single_user_df[0])

def prune_plist(pf_list,min_freq,max_per) :
    for key in pf_list.keys() :
        if pf_list[key]['per'] > max_per or pf_list[key]['freq'] < min_freq :
        del pf_list[key]
    print pf_list
    return pf_list

prune_plist(plist(single_user_df[0]),3,4)