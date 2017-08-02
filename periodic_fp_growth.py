import itertools

import os
import time
# os.chdir('/root/mb/market_basket_data')
os.chdir('/home/prudhvi/Documents/market_basket_data')
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



# frequent items with key value pairs required


class FPTree(object):
    """
    A frequent pattern tree.
    """

    def __init__(self, transactions,root_value, root_count):
        """
        Initialize the tree.
        """
        #self.frequent = self.find_frequent_items(transactions, threshold)
        self.headers = self.build_header_table(self.frequent)
        self.root = self.build_fptree(transactions, root_value,
            root_count, self.frequent, self.headers)

    def build_header_table(frequent):
        """
        Build the header table.
        """
        headers = {}
        for key in frequent.keys():
            headers[key] = None

        return headers

    def build_fptree(self, transactions, root_value, root_count, frequent, headers):
        """
        Build the FP tree and return the root node.
        """
        root = FPNode(root_value, root_count, None)

        for transaction in transactions:
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: frequent[x], reverse=True)
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items, root, headers)

        return root

    def insert_tree(self, items, node, headers):
        """
        Recursively grow FP tree.
        """
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
        else:
            # Add new child.
            child = node.add_child(first)

            # Link it to header structure.
            if headers[first] is None:
                headers[first] = child
            else:
                current = headers[first]
                while current.link is not None:
                    current = current.link
                current.link = child

        # Call function recursively.
        remaining_items = items[1:]
        if len(remaining_items) > 0:
            self.insert_tree(remaining_items, child, headers)


    def tree_has_single_path(self, node):
        """
        If there is a single path in the tree,
        return True, else return False.
        """
        num_children = len(node.children)
        if num_children > 1:
            return False
        elif num_children == 0:
            return True
        else:
            return True and self.tree_has_single_path(node.children[0])





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

single_user_df = prior_with_userids[prior_with_userids['user_id']==1]

userids_with_orderlist = single_user_df.groupby(['user_id','order_id'])['product_id','order_number'].apply(lambda x: x['product_id'].tolist()).reset_index()
qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq

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