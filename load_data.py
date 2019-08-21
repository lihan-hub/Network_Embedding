import pandas as pd
from collections import OrderedDict

data_path = "hmdd2/alldata.xlsx"
raw_links = pd.read_excel(data_path, header=None, usecols=[1, 2])
raw_links.columns = ["miRNA", "disease"]
print(raw_links.tail(6))
print(raw_links.astype(str).describe(include='all'))
raw_links.drop_duplicates(inplace=True)
raw_links.reset_index(drop=True, inplace=True)
print(raw_links.tail(6))
print(raw_links.astype(str).describe(include='all'))
raw_links.sort_values(by=["miRNA", "disease"], ascending=[True, True], inplace=True)
raw_links.reset_index(drop=True, inplace=True)
raw_links.to_csv("filtered_MD_links.txt", sep="\t", header=None, index=True)
print(raw_links.head())
# 过滤数据，删除disease-miRNA的孤立边
miRNA_links_counts = raw_links["miRNA"].value_counts()
print(miRNA_links_counts)
miRNA_has_one_link = miRNA_links_counts[miRNA_links_counts == 1].index.tolist()     # 找到只有一条边的miRNA
print(miRNA_has_one_link)

disease_links_counts = raw_links["disease"].value_counts()
print(disease_links_counts)
disease_has_one_link = disease_links_counts[disease_links_counts == 1].index.tolist()
print(disease_has_one_link)
ix2filter = raw_links[raw_links["miRNA"].isin(miRNA_has_one_link)].index.tolist()
print(ix2filter)

for ix in ix2filter:
    disease_temp = raw_links.iloc[ix, 1]
    if disease_temp in disease_has_one_link:
        print("需移除数据:", ix, raw_links.iloc[ix, 0], disease_temp)
        # 删除原索引为748行的Glomerulonephritis疾病，及对应hsa-mir-1207 miRNA
        raw_links.drop(ix, inplace=True)
# 重新写入文件
raw_links.reset_index(drop=True, inplace=True)
raw_links.to_csv("filtered_MD_links.txt", sep="\t", header=None, index=True)
