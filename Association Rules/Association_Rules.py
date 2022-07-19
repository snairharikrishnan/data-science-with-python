import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sklearn
from collections import Counter
from mlxtend.frequent_patterns import apriori,association_rules

#with open("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\my_movies.csv") as f:
#    movies=f.read()

movies=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\my_movies.csv")
movies_dummy=movies.iloc[:,5:]
movies_list=[]
for i in range(5):
    list1=list(movies.iloc[i,:5])
    one_transaction=[i for i in list1 if type(i)==str]    
    movies_list.append(one_transaction)
        
all_movies=[i for item in movies_list for i in item]
item_frequency=Counter(all_movies)        
        
item_frequency=sorted(item_frequency.items(),key=lambda x:x[1])

item=list(reversed([i[0] for i in item_frequency]))        
frequency=list(reversed([i[1] for i in item_frequency]))

plt.bar(height=frequency,x=item,color = 'rgbkymc')

frequent_items=apriori(movies_dummy,min_support=0.05,max_len=3,use_colnames=True)
#most frequent based on support
frequent_items.sort_values('support',ascending=False,inplace=True)
plt.bar(height=frequent_items.support[0:11],x=np.arange(11));plt.xticks(np.arange(11),frequent_items.itemsets[1:11])

rules=association_rules(frequent_items,metric='lift',min_threshold=1)
rules.sort_values('lift',ascending=False,inplace=True)
rules.head(10).iloc[:,[0,1,6]]
print(rules.head(10).iloc[:,[0,1,6]])

# To prevent profusion of rules
def to_list(i):
    return(sorted(list(i)))

X=list(rules.antecedents.apply(to_list)+rules.consequents.apply(to_list))
    
for i in X:
    i.sort()
    
unique_rules=[list(m) for m in set(tuple(i) for i in X)]
unique_index=[]
for i in unique_rules:
    unique_index.append(X.index(i))

rules_no_prof=rules.iloc[unique_index,:]
rules_no_prof.sort_values('lift',ascending=False,inplace=True)
rules_no_prof.head(10).iloc[:,[0,1,6]]
