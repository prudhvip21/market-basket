import itertools

import os
import time
# os.chdir('/root/mb/market_basket_data')
os.chdir('/home/prudhvi/Documents/market_basket_data')
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score, confusion_matrix
import numpy as np # linear algebra
import pandas as pd
import  anytree
from anytree import  RenderTree
from copy import deepcopy


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
        self.name = value
        self.count = count
        self.parent = parent
        self.link = None
        self.children = []
        self.transactions = []
        self.flag = 0

    def has_child(self, value):
        """
        Check if node has a particular child node.
        """
        for node in self.children:
            if node.name == value:
                return True

        return False

    def get_child(self, value):
        """
        Return a child node with a particular value.
        """
        for node in self.children:
            if node.name == value:
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
    def __init__(self, transactions,frequent,root_value, root_count):
        """
        Initialize the tree.
        """
        self.frequent = frequent
        self.headers = build_header_table(frequent)
        self.root = self.build_fptree(transactions, root_value,
            root_count, self.frequent, self.headers)

    def __repr__(self):
        return 'node(' + repr(self.root.value) + ', ' + repr(self.root.children) + ')'

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

        for ind,transaction in enumerate(transactions):
            sorted_items = [x for x in transaction if x in frequent]
            sorted_items.sort(key=lambda x: (frequent[x],x), reverse=True)
            if len(sorted_items) > 0:
                self.insert_tree(sorted_items,ind,root, headers)

        return root

    def insert_tree(self, items,ind,node, headers):
        """
        Recursively grow FP tree.
        """
        first = items[0]
        child = node.get_child(first)
        if child is not None:
            child.count += 1
            if len(items) == 1:
                child.transactions.append(ind)
        else:
            # Add new child.
            child = node.add_child(first)
            if len(items) == 1:
                child.transactions.append(ind)
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
            self.insert_tree(remaining_items,ind,child, headers)

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


# single_user_df = pd.merge(singleuser_with_orderlist,orders_df.iloc[:,[0,3]],on = 'order_id' , how = 'left')



def prune_plist(pf_list) :
    freqs = [pf_list[key]['freq'] for key in pf_list.keys()]
    min_freq = np.percentile(freqs,20)
    pers =  [pf_list[key]['per'] for key in pf_list.keys()]
    max_per = np.percentile(pers,80)
    for key in pf_list.keys() :
        if pf_list[key]['per'] > max_per or pf_list[key]['freq'] < min_freq :
            del pf_list[key]

    for key in pf_list.keys() :
        pf_list[key] = pf_list[key]['freq']

    #print pf_list
    return pf_list

freq = prune_plist(plist(singleuser_with_orderlist[0]))

# fp_tree

#freq = build_header_table(freq)



def prune_tree(temp_tree,node_value) :
    tree = deepcopy(temp_tree)
    current = tree.headers[node_value]
    while current.link is not None:
        temp = current
        while temp.name is not 0:
            #print "1"
            temp.flag = 1
            #print temp.flag
            temp = temp.parent

        if current.parent.name != 0 :
            current.parent.transactions.extend(current.transactions)
        current.parent.children.remove(current)
        current = current.link

    temp = current
    while temp.name is not 0:
        temp.flag = 1
        temp = temp.parent

    if current.parent.name != 0:
        current.parent.transactions.extend(current.transactions)
    current.parent.children.remove(current)

    for pre,fill,node in RenderTree(tree.root):
        if node.name != 0 and node.flag == 0:
            node.parent.children.remove(node)

    for pre, fill,node in RenderTree(tree.root):
        # print pre
        # node.flag = 0
        if len(node.transactions) != 0:
            temp = node
            while temp.parent is not None:
                if temp.parent.name != 0 :
                    temp.parent.transactions.extend(temp.transactions)
                    temp.parent.transactions = list(set(temp.parent.transactions))

                temp = temp.parent

    return tree




for pre,fill,node in RenderTree(fptree1.root):
    print("%s%s" % (pre,node.name))


def conditional_patterns(tree_pruned,pattern_node,prns) :

    for pre, fill, node in RenderTree(tree_pruned.root):
        if  node.name is not 0 :
            try :
                trns = node.transactions
                #print trns
                k = [(trns[i + 1] - trns[i]) for i in range(len(trns)) if i <= len(trns) - 2]
                #print k
                per = max(k)
                f = len(trns)
                pattern = str(pattern_node) + ","+ str(node.name)
                if per < 7 and f > 2 :
                    prns[pattern] = [f,per]
                    #print pattern
            except :
                 pass
    return prns


def next_pftree(original_tree,node) :
    tem = deepcopy(original_tree)
    n = tem.headers[node]
    while True :
        n.parent.transactions.extend(n.transactions)
        n.parent.children.remove(n)
        if n.link is None :
            break
        else :
            n = n.link
    return tem

def generate_patterns(transaction_list,transactions) :
    freq = prune_plist(transactions)
    fptree  = FPTree(transaction_list, freq, 0, 0)
    pf_table = freq.items()
    pf_table.sort(key = operator.itemgetter(1,0))
    patterns = { }
    prns = {}
    for item in pf_table :
        fptree_pruned = prune_tree(fptree, item[0])
        pat = conditional_patterns(fptree_pruned,item[0],prns)
        patterns.update(pat)
        fptree = next_pftree(fptree,item[0])

    return patterns


orders_df_test = orders_df[orders_df['eval_set']=='test']
userids_list = list(set(orders_df_test['user_id']))
prior_with_userids = pd.merge(order_products_prior_df, orders_df, on='order_id', how='left')

del orders_df
del order_products_prior_df
del prior_with_userids


def final_submission(prior,orders_df,d_min,userids_list) :
    i = 0
    submiss = {}
    for z in userids_list :
        i = i + 1
        try :
            single_user_df = prior[prior['user_id']==z]
            single_user_df = single_user_df.sort_values(by ='order_number')
            singleuser_with_orderlist = single_user_df.groupby(['user_id','order_id'])['product_id','order_number'].apply(lambda x: x['product_id'].tolist()).reset_index()
            final_df = pd.merge(singleuser_with_orderlist,orders_df,on =['order_id','user_id'], how= 'left')
            transaction_list = final_df[0].tolist()
            transactions = plist(final_df[0])
            patrns= generate_patterns(transaction_list,transactions)
            pm = p_min(final_df, patrns)
            rated_items = tbp_predictor(final_df,patrns,d_min,pm)
            predicted_list = final_product_list(final_df,rated_items)
            print z
            print predicted_list
            submiss[z] = predicted_list
            if i > 20 :
                break

        except :
            submiss[z] = ' '
            pass

        print i ,"users predicted"
    return submiss


kk = final_submission(prior_with_userids,orders_df_test,5,userids_list)



sub = pd.DataFrame(kk.items(), columns=['user_id', 'Products'])


final = pd.merge(orders_df_test,sub,on = 'user_id' , how = 'outer')


def flatten(x) :
    try :
        if len(x) == 0:
            return ' '
        else  :
            return " ".join(str(i) for i in x)
    except :
        return  ' '


final['Products'] = final['Products'].apply(flatten)
final.to_csv( path_or_buf ="~/sub.csv", header = True )
prior = prior_with_userids



"""Test for one user"""


sub = pd.read_csv("/home/prudhvi/Dropbox/MB_project/market-basket/sub.csv")

sub = pd.merge(sub,orders_df_test,on = 'order_id' , how = 'left')

sub.to_csv(path_or_buf ="~/sub.csv", header = True )



single_user_df = prior_with_userids[prior_with_userids['user_id'] == 12]
single_user_df = single_user_df.sort_values(by='order_number')

singleuser_with_orderlist = single_user_df.groupby(['user_id','order_id'])['product_id','order_number'].apply(
    lambda x: x['product_id'].tolist()).reset_index()



final_df = pd.merge(singleuser_with_orderlist, orders_df, on=['order_id', 'user_id'], how='left')
transaction_list = final_df[0].tolist()
transactions = plist(final_df[0])
patrns = generate_patterns(transaction_list, transactions)
pm = p_min(final_df, patrns)
rated_items = tbp_predictor(final_df, patrns, 5, pm)
predicted_list = final_product_list(final_df, rated_items)































"""Junk Code   



for item in pat.items() :

l = [item[0].split(',') for item in pat.items()]

l = [item for sublist in l for item in sublist]


node = fptree1.headers[26088]
for i in range(5):
    print node.name
    node = node.link


def node_recurse_generator(node):
    yield node.value
    #print node.value
    for n in node.children:
        for rn in node_recurse_generator(n):
            yield rn
        #yield "EnD"


list(node_recurse_generator(fptree1.root))

transaction = singleuser_with_orderlist[0].tolist()[2]
t = [x for x in transaction if x in freq]

t.sort(key=lambda x: (freq[x],x), reverse=True)

for index,value in enumerate(i) :
    print index  


    current = fptree1.headers[13032]


current = fptree1.headers[13032]
for i in range(5)  :
    if current.link is None :
        print current.parent.name
        break
    else :
        print current.parent.name
        current = current.link


                temp = pd.DataFrame([pattern,f,per],columns =('pattern','frequency','periodicity'))
                df.append(temp)
                #df.loc[len(df)] = pattern 
                
                
                trns = fptree1_pruned.root.children[0].transactions

k = [(trns[i+1]-trns[i]) for i in range(len(trns)) if i <= len(trns)-2]


import operator
sorted_x = sorted(freq.items(), key= lambda x: (operator.itemgetter(1),operator.itemgetter(0)),reverse = True)


i = freq.items() 

kk = next_pftree(fptree1,26088)



conditional_patterns(fptree1_pruned,13176)
 
 

fptree1_pruned = prune_tree(fptree1,13176) 




fptree1 = FPTree(singleuser_with_orderlist[0].tolist(),freq,0,0)


plist(singleuser_with_orderlist[0])


for pre,fill,node in RenderTree(tree.root):
    #print pre
    #node.flag = 0
    print("%s%s" % (pre,node.name))

"""