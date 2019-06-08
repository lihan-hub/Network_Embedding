# -*- coding: utf-8 -*-
# coding: utf-8
"""
This version can be used to reproduce what have been done.
"""
from collections import OrderedDict
#######################################################################################################
#######################################################################################################
DISEASE_GENE_FILE = "datasets/DG.txt"
MIRNA_DISEASE_FILE = "datasets/MD.txt"
WRITE_PATH = "datasets/"
# 存储数据为 id_gene.txt、id_disease、id_miRNA.txt、
# disease_gene.txt、disease_miRNA.txt(Note:这两个文件全是用上述三个文件中的id表示的)

gene_id = OrderedDict()
disease_id = OrderedDict()
miRNA_id = OrderedDict()
disease_gene = OrderedDict()
disease_miRNA = OrderedDict()

gene_names = []
disease_names_of_gene = []
with open(DISEASE_GENE_FILE) as tsv:
    for line in tsv:
        toks = line.strip().split()
        disease_temp = " ".join(toks[:-1])
        gene_temp = toks[-1]
        if len(toks) >= 2:
            disease_names_of_gene.append(disease_temp)
            gene_names.append(gene_temp)
        # added
        if disease_temp not in disease_gene:
            disease_gene[disease_temp] = []
        disease_gene[disease_temp].append(gene_temp)

    gene_names_set = list(set(gene_names))
    disease_names_of_gene_unique = list(set(disease_names_of_gene))
    disease_names_of_gene_unique.sort(key=disease_names_of_gene.index)

miRNA_names = []
disease_names_of_miRNA = []
with open(MIRNA_DISEASE_FILE) as tsv:
    for line in tsv:
        toks = line.strip().split()
        disease_temp = " ".join(toks[1:])
        miRNA_temp = toks[0]
        if len(toks) >= 2:
            disease_names_of_miRNA.append(disease_temp)
            miRNA_names.append(miRNA_temp)
        # added
        if disease_temp not in disease_miRNA:
            disease_miRNA[disease_temp] = []
        disease_miRNA[disease_temp].append(miRNA_temp)
    miRNA_names_set = list(set(miRNA_names))
    disease_names_of_miRNA_unique = list(set(disease_names_of_miRNA))
    disease_names_of_miRNA_unique.sort(key=disease_names_of_miRNA.index)

print("original: total edges of gene-disease data: ", len(disease_names_of_gene))
print("original: total unique gene of disease-gene data: ", len(gene_names_set))
print("original: gene-associated disease: ", len(disease_names_of_gene_unique))

print("\noriginal: total edges of disease-miRNA data: ", len(disease_names_of_miRNA))
print("original: miRNA-associated disease: ", len(disease_names_of_miRNA_unique))
print("original: total unique miRNA of disease-miRNA data: ", len(miRNA_names_set))

#######################################################################################################
#######################################################################################################
# 求并集
disease_name_set = []
for disease_of_gene in disease_names_of_gene_unique:
    if disease_of_gene not in disease_name_set:
        disease_name_set.append(disease_of_gene)
for disease_of_miRNA in disease_names_of_miRNA_unique:
    if disease_of_miRNA not in disease_name_set:
        disease_name_set.append(disease_of_miRNA)

# 以字典形式存储所有disease的name和id
for index in range(0, len(disease_name_set)):
    disease_id[disease_name_set[index]] = 'D-'+str(index)

miRNA_names_set.sort(key=miRNA_names.index)     # 对列表去重，并保持原有的顺序 for reproducing
# 已字典形式存储miRNA的name和id
for index in range(0, len(miRNA_names_set)):
    miRNA_id[miRNA_names_set[index]] = 'mi-'+str(index + len(disease_name_set))

gene_names_set.sort(key=gene_names.index)       # 对列表去重，并保持原有的顺序 for reproducing
# 已字典形式存储gene的name和id
for index in range(0, len(gene_names_set)):
    gene_id[gene_names_set[index]] = 'G-'+str(index + len(disease_name_set) + len(miRNA_names_set))

#######################################################################################################
#######################################################################################################
# 保存至 .txt文件
outfile = open(WRITE_PATH + "id_gene.txt", 'w')
for gene, id_number in gene_id.items():
    outline = id_number + "\t" + 'Gene-' + gene.lower()
    outfile.write(outline + "\n")
outfile.close()

outfile = open(WRITE_PATH + "id_disease.txt", 'w')
for disease, id_number in disease_id.items():
    outline = id_number + "\t" + 'Disease-' + disease.lower()
    outfile.write(outline + "\n")
outfile.close()

outfile = open(WRITE_PATH + "id_miRNA.txt", 'w')
for miRNA, id_number in miRNA_id.items():
    outline = id_number + "\t" + 'MiRNA-' + miRNA.lower()
    outfile.write(outline + "\n")
outfile.close()

outfile = open(WRITE_PATH + "disease_gene.txt", 'w')
for disease, gene in zip(disease_names_of_gene, gene_names):
    outline = disease_id[disease] + "\t" + gene_id[gene]
    outfile.write(outline + "\n")
outfile.close()

outfile = open(WRITE_PATH + "disease_miRNA.txt", 'w')
# sorted by disease name for better understanding
for disease, miRNA in sorted(zip(disease_names_of_miRNA, miRNA_names)):
    outline = disease_id[disease] + "\t" + miRNA_id[miRNA]
    outfile.write(outline + "\n")
outfile.close()

# write node's adjacent list to .txt file
outfile = open(WRITE_PATH + "disease_miRNA_node_adjlist.txt", 'w')
node_adjlist = OrderedDict()
with open(WRITE_PATH + "disease_miRNA.txt") as tsv:
    for line in tsv:
        toks = line.strip().split("\t")
        node1 = toks[0]
        node2 = toks[1]
        if node1 not in node_adjlist:
            node_adjlist[node1] = []
        node_adjlist[node1].append(node2)

for node, node_adj in node_adjlist.items():
    outline = node
    for adj in node_adj:
        outline += " " + adj
    outfile.write(outline + "\n")
outfile.close()

# 测试用
# 选取节点邻居较多的前n个节点对应的键-值
adj_length = [(len(v), k) for k, v in node_adjlist.items()]
top_k = 30
length_key_ls = sorted(adj_length, reverse=True)[:top_k]       # 降序
print(length_key_ls)
top_node_adjlist = OrderedDict()
for length_key in length_key_ls:
    key = length_key[1]
    top_node_adjlist[key] = node_adjlist[key]

outfile_name = "top-{} disease_miRNA_node_adjlist.txt".format(str(top_k))
outfile = open(WRITE_PATH + outfile_name, 'w')
for node in top_node_adjlist:
    node_adj = top_node_adjlist[node]
    outline = node
    for adj in node_adj:
        outline += " " + adj
    outfile.write(outline + "\n")
outfile.close()

# added, 不同算法所用数据格式
#######################################################################################################
'''
                    README
                    
node        coding rule                 total
disease     D-0     ==>     D-329       330
miRNA       mi-330  ==>     mi-821      492
gene        G-822   ==>     G-5241      4420

'''
#######################################################################################################
# data saved for deep_walk training
'''
deepwalk 只接受edgelist中编码为数字, eg:
1 2
1 3
2 4
因此此处需要做id-mapping，原始数据中：
疾病已编码为D-number形式
miRNA已编码为mi-number形式
基因已编码为G-number形式(暂未用到)
'''
import os
import shutil
deepwalk_path = "datasets/deepwalk/"
if not os.path.exists(deepwalk_path):
    os.mkdir(deepwalk_path)
output_links_name = deepwalk_path + "deepwalk_needed_links.txt"
output_links_file = open(output_links_name, 'w')
for disease in disease_id:
    try:
        for miRNA_temp in disease_miRNA[disease]:
            disease_str_id = disease_id[disease]
            miRNA_str_id = miRNA_id[miRNA_temp]
            disease_num_id = disease_str_id.split("-")[1]
            miRNA_num_id = miRNA_str_id.split("-")[1]       # convert formats to deep_walk needed
            outline = disease_num_id + "\t" + miRNA_num_id
            output_links_file.write(outline + "\n")
        for gene_temp in disease_gene[disease]:
            disease_str_id = disease_id[disease]
            gene_str_id = gene_id[gene_temp]
            disease_num_id = disease_str_id.split("-")[1]
            gene_num_id = gene_str_id.split("-")[1]         # convert formats to deep_walk needed
            outline = disease_num_id + "\t" + gene_num_id
            output_links_file.write(outline + "\n")
    except KeyError:
        pass
output_links_file.close()
# copy and rename file to node2vec directory
node2vec_path = "datasets/node2vec/"
node2vec_file = node2vec_path + "node2vec_needed_links.txt"
shutil.copy(output_links_name, node2vec_file)


# data saved for line training
'''
line 输入数据格式（权重可以不要）:
D-0	    mi-268
mi-268	D-0
D-1	    mi-268
mi-268	D-1
因此此处需要做id-mapping，原始数据中：
疾病已编码为D-number形式
miRNA已编码为mi-number形式
基因已编码为G-number形式(暂未用到)
'''
line_path = "datasets/line/"
if not os.path.exists(line_path):
    os.mkdir(line_path)
output_links_name = line_path + "line_needed_links.txt"
output_links_file = open(output_links_name, 'w')
for disease in disease_id:
    try:
        for miRNA_temp in disease_miRNA[disease]:
            '''
            # 编码为数字id，会丢失部分节点向量
            disease_str_id = disease_id[disease]
            miRNA_str_id = miRNA_id[miRNA_temp]
            disease_num_id = disease_str_id.split("-")[1]
            miRNA_num_id = miRNA_str_id.split("-")[1]       # convert formats to deep_walk needed
            outline = disease_num_id + "\t" + miRNA_num_id + "\n"
            outline += miRNA_num_id + "\t" + disease_num_id
            output_links_file.write(outline + "\n")
            '''
            disease_str_id = disease_id[disease]
            miRNA_str_id = miRNA_id[miRNA_temp]
            outline = disease_str_id + "\t" + miRNA_str_id + "\n"
            outline += miRNA_str_id + "\t" + disease_str_id
            output_links_file.write(outline + "\n")
        for gene_temp in disease_gene[disease]:
            disease_str_id = disease_id[disease]
            gene_str_id = gene_id[gene_temp]
            outline = disease_str_id + "\t" + gene_str_id + "\n"
            outline += gene_str_id + "\t" + disease_str_id
            output_links_file.write(outline + "\n")
    except KeyError:
        pass
output_links_file.close()
