#######################################################################################################
#######################################################################################################
DISEASE_GENE_FILE = "datasets/DG.txt"
MIRNA_DISEASE_FILE = "datasets/MD.txt"
outfilename = "id_gene.txt"

# 存储数据为 id_gene.txt、id_disease、id_miRNA.txt、
# disease_gene.txt、disease_miRNA.txt(Note:这两个文件全是用上述三个文件中的id表示的)
gene_id = {}
disease_id = {}
miRNA_id = {}
disease_gene = {}
disease_miRNA = {}

gene_names = []
disease_names_of_gene = []
with open(DISEASE_GENE_FILE) as tsv:
    for line in tsv:
        toks = line.strip().split()
        if len(toks) >= 2:
            disease_names_of_gene.append(" ".join(toks[:-1]))
            gene_names.append(toks[-1])

    gene_names = list(set(gene_names))
    disease_names_of_gene = list(set(disease_names_of_gene))


miRNA_names = []
disease_names_of_miRNA = []
with open(MIRNA_DISEASE_FILE) as tsv:
    for line in tsv:
        toks = line.strip().split()
        if len(toks) >= 2:
            disease_names_of_miRNA.append(" ".join(toks[1:]))
            miRNA_names.append(toks[0])

    miRNA_names = list(set(miRNA_names))
    disease_names_of_miRNA = list(set(disease_names_of_miRNA))

print("disease_names_of_gene: ", len(disease_names_of_gene))
print("disease_names_of_miRNA: ", len(disease_names_of_miRNA))

# 求交集，使得disease在两个文件中都有数据存在
disease_name_set = [disease for disease in disease_names_of_gene if disease in disease_names_of_miRNA]
# 已字典形式存储disease的name和id
for index in range(0, len(disease_name_set)):
    disease_id[disease_name_set[index]] = 'D-'+str(index)

#######################################################################################################
#######################################################################################################
# 重新载入数据：只保留与disease_name相关的gene和miRNA对应的行
gene_names = []
disease_names_of_gene = []
with open(DISEASE_GENE_FILE) as tsv:
    disease_temp = ""
    gene_temp = ""
    for line in tsv:
        toks = line.strip().split()
        if len(toks) >= 2:
            disease_temp = " ".join(toks[:-1])
            gene_temp = toks[-1]
        if disease_temp in disease_name_set:
            disease_names_of_gene.append(disease_temp)
            gene_names.append(gene_temp)

gene_names_set = list(set(gene_names))
# 已字典形式存储gene的name和id
for index in range(0, len(gene_names_set)):
    gene_id[gene_names_set[index]] = 'G-'+str(index)

miRNA_names = []
disease_names_of_miRNA = []
with open(MIRNA_DISEASE_FILE) as tsv:
    disease_temp = ""
    miRNA_temp = ""
    for line in tsv:
        toks = line.strip().split()
        if len(toks) >= 2:
            disease_temp = " ".join(toks[1:])
            miRNA_temp = toks[0]
        if disease_temp in disease_name_set:
            disease_names_of_miRNA.append(disease_temp)
            miRNA_names.append(miRNA_temp)

miRNA_names_set = list(set(miRNA_names))
# 已字典形式存储miRNA的name和id
for index in range(0, len(miRNA_names_set)):
    miRNA_id[miRNA_names_set[index]] = 'mi-'+str(index)

#######################################################################################################
#######################################################################################################
# 保存至 .txt文件
outfile = open("id_gene.txt", 'w')
for gene, id_number in gene_id.items():
    outline = id_number + "\t" + 'Gene-' + gene.lower()
    outfile.write(outline + "\n")
outfile.close()

outfile = open("id_disease.txt", 'w')
for disease, id_number in disease_id.items():
    outline = id_number + "\t" + 'Disease-' + disease.lower()
    outfile.write(outline + "\n")
outfile.close()

outfile = open("id_miRNA.txt", 'w')
for miRNA, id_number in miRNA_id.items():
    outline = id_number + "\t" + 'MiRNA-' + miRNA.lower()
    outfile.write(outline + "\n")
outfile.close()

outfile = open("disease_gene.txt", 'w')
for disease, gene in zip(disease_names_of_gene, gene_names):
    outline = disease_id[disease] + "\t" + gene_id[gene]
    outfile.write(outline + "\n")
outfile.close()

outfile = open("disease_miRNA.txt", 'w')
for disease, miRNA in zip(disease_names_of_miRNA, miRNA_names):
    outline = disease_id[disease] + "\t" + miRNA_id[miRNA]
    outfile.write(outline + "\n")
outfile.close()
