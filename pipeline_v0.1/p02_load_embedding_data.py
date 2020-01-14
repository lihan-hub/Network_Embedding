# 准备disease-miRNA pairs 正样本2101对
# leave-one-out cross-validation
import pandas as pd
import sys
import matplotlib.pyplot as plt

id_disease = {}     # 143 disease
id_miRNA = {}       # 393 miRNA
with open("datasets/id_disease.txt") as tsv:
    disease_temp = ""
    id_temp = ""
    for line in tsv:
        toks = line.strip().split("\t")
        id_temp = toks[0]
        disease_temp = toks[1].replace(" ", "")
        id_disease[id_temp] = disease_temp

with open("datasets/id_miRNA.txt") as tsv:
    miRNA_temp = ""
    id_temp = ""
    for line in tsv:
        toks = line.strip().split("\t")
        id_temp = toks[0]
        miRNA_temp = toks[1]
        id_miRNA[id_temp] = miRNA_temp


disease_name = []
miRNA_name = []
miRNA_disease_dic = {}
with open("datasets/disease_miRNA.txt") as tsv:
    disease_temp = ""
    miRNA_temp = ""
    for line in tsv:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            disease_temp = toks[0]
            miRNA_temp = toks[1]
        name_miRNA = id_miRNA[miRNA_temp]
        name_disease = id_disease[disease_temp]

        disease_name.append(name_disease)
        miRNA_name.append(name_miRNA)

        if name_miRNA not in miRNA_disease_dic:
            miRNA_disease_dic[name_miRNA] = []
        # 方便后续移除一个miRNA时，移除对应的disease
        miRNA_disease_dic[name_miRNA].append(name_disease)

disease_vector = {}
miRNA_vector = {}
# change following path to run different vectors
# VECTOR_PATH = "datasets/gene-disease-miRNA.mdgdm.w100.l1000.pp1.size64.window7.negative5.txt"
VECTOR_PATH = "output_vectors/gene-disease-miRNA.mdgdm.w1000.l100.pp1.size32.window7.negative5.txt.txt"
# VECTOR_PATH = "output_vectors/gene-disease-miRNA.mdgdm.w1000.l100.pp0.size64.window7.negative5.txt.txt"
# VECTOR_PATH = "output_vectors/gene-disease-miRNA.mdgdm.w200.l200.pp1.size128.window7.negative5.txt.txt"
with open(VECTOR_PATH) as tsv:
    next(tsv)
    next(tsv)
    for line in tsv:
        toks = line.strip().split()
        node_temp = toks[0]
        vector_temp = toks[1:]
        vector_temp_int = list(map(float, vector_temp))
        if node_temp.startswith("Disease"):
            disease_vector[node_temp] = vector_temp_int
        elif node_temp.startswith("MiRNA"):
            miRNA_vector[node_temp] = vector_temp_int
"""
disease_vector_df = pd.DataFrame.from_dict(disease_vector, orient='index')
disease_vector_df = disease_vector_df.reset_index().rename(columns={'index': 'disease'})
miRNA_vector_df = pd.DataFrame.from_dict(miRNA_vector, orient='index')
miRNA_vector_df = miRNA_vector_df.reset_index().rename(columns={'index': 'miRNA'})
"""
#######################################################################################################
#######################################################################################################
import numpy as np


# 特征组合方式
def feature_combination_Hadamard(feature_vec1, feature_vec2):
    return list(np.multiply(feature_vec1, feature_vec2))


def feature_combination_Concatenate(feature_vec1, feature_vec2):
    return list(np.r_[feature_vec1, feature_vec2])


def feature_combination_Average(feature_vec1, feature_vec2):
    return list((feature_vec1 + feature_vec2) / 2)


# 比例不同，效果也不一样。
"""
# a.feature_combination_Concatenate:
## w1000.l100.pp1.size32    64维
neg_ratio:  0.1     0.15    0.2     0.25    0.3     0.35    0.4     0.45    0.5     0.55    0.6     0.7     0.8
AUC:        0.836   0.868   0.850   0.866   0.862   0.851   0.841   0.828   0.795   0.771   0.746   0.623   0.620
## w1000.l100.pp1.size64    128维
neg_ratio:  0.1     0.15    0.2     0.25    0.3     0.35    0.4     0.45    0.5     0.55    0.6     0.7     0.8
AUC:        0.752   0.823   0.836   0.852   0.825   0.816   0.797   0.782   0.688   0.660   0.589   0.555   0.562

# b.feature_combination_Hadamard:
## w1000.l100.pp1.size32    32维
neg_ratio:  0.1     0.15    0.2     0.25    0.3     0.35    0.4     0.45    0.5     0.55    0.6     0.7     0.8
AUC:        0.722   0.741   0.752   0.736   0.739   0.742   0.732   0.731   0.723   0.722   0.712   0.713   0.712
## w1000.l100.pp1.size64    64维
neg_ratio:  0.1     0.15    0.2     0.25    0.3     0.35    0.4     0.45    0.5     0.55    0.6     0.7     0.8
AUC:        0.768   0.758   0.775   0.795   0.789   0.780   0.788   0.783   0.772   0.773   0.752   0.745   0.741                  
"""
# "二级排序"——第一次排序根据字符串中的字母，第二次根据字符串末尾的数字
# 备份字v2.0版
def get_sorted_data(link_data):
    print("# sorting link data")
    keys = sorted(link_data.keys())
    ix_pre = 0
    while ix_pre < len(link_data) - 1:      # 之前用for循环，keys[ix_pre]并未更新，改用while
        for ix_after, key_after in enumerate(keys[ix_pre + 1:], start=ix_pre + 1):
            key_pre = keys[ix_pre]
            entity_pre = key_pre.split("::")[0]
            num_pre = key_pre.split("-")[2]

            entity_after = key_after.split("::")[0]
            num_after = key_after.split("-")[2]
            if entity_pre == entity_after:
                if int(num_after) < int(num_pre):
                    temp = keys[ix_pre:ix_after]
                    keys[ix_pre + 1:ix_after + 1] = temp
                    keys[ix_pre] = key_after
            else:
                break
        ix_pre = ix_pre + 1

    temp_dict = link_data.copy()
    del link_data
    link_data = {}
    for key in keys:
        link_data[key] = temp_dict[key]
    return link_data


# 生成负样本
def get_neg_data(pos_data, method="1", neg_ratio=0.2, feature_combination=feature_combination_Concatenate):
    miRNA_disease_neg = {}
    np.random.seed(22)      # for reproduce
    shuffled_indices = np.random.permutation(len(disease_name))

    if method == '1':
        # 负样本产生方式一
        for index in shuffled_indices:
            if len(miRNA_disease_neg) == int(len(pos_data) * neg_ratio):
                break
            neg_link = miRNA_name[index] + "::" + disease_name[-index]
            if neg_link in pos_data:
                continue
            if neg_link not in miRNA_disease_neg:
                miRNA_disease_neg[neg_link] = []
            miRNA_name_temp = miRNA_name[index]
            disease_name_temp = disease_name[-index]
            # scale features
            miRNA_feature = np.array(miRNA_vector[miRNA_name_temp]) / np.array(miRNA_vector[miRNA_name_temp]).sum()
            disease_feature = np.array(disease_vector[disease_name_temp]) / np.array(disease_vector[disease_name_temp]).sum()
            miRNA_disease_neg[neg_link] = feature_combination(miRNA_feature, disease_feature)
        return miRNA_disease_neg

    if method == "2":
        # 负样本产生方式二
        indices = np.random.choice(shuffled_indices, int(len(pos_data) * neg_ratio))
        while len(miRNA_disease_neg) < len(indices):
            ix_1 = np.random.choice(indices)
            ix_2 = np.random.choice(indices)
            if ix_1 == ix_2:
                continue
            neg_link = miRNA_name[ix_1] + "::" + disease_name[ix_2]
            if neg_link in pos_data:
                continue
            if neg_link not in miRNA_disease_neg:
                miRNA_disease_neg[neg_link] = []
            miRNA_name_temp = miRNA_name[ix_1]
            disease_name_temp = disease_name[ix_2]
            # scale features
            miRNA_feature = np.array(miRNA_vector[miRNA_name_temp]) / np.array(miRNA_vector[miRNA_name_temp]).sum()
            disease_feature = np.array(disease_vector[disease_name_temp]) / np.array(disease_vector[disease_name_temp]).sum()
            miRNA_disease_neg[neg_link] = feature_combination(miRNA_feature, disease_feature)
        return miRNA_disease_neg


# 计算 miRNA-disease pairs 的相乘特征：64维
# 预测 与 miRNA 相关的疾病
# 生成正样本
def get_pos_data(feature_combination=feature_combination_Concatenate, neg_ratio=0.2):
    vector_temp_int = list(map(float, vector_temp))
    miRNA_disease_pos = {}
    for miRNA in miRNA_disease_dic:
        for disease in miRNA_disease_dic[miRNA]:
            pos_link = miRNA + "::" + disease
            if pos_link not in miRNA_disease_pos:
                miRNA_disease_pos[pos_link] = []
            miRNA_feature = np.array(miRNA_vector[miRNA])
            disease_feature = np.array(disease_vector[disease])
            miRNA_disease_pos[pos_link] = feature_combination(miRNA_feature, disease_feature)

    """
    # 生成负样本
    miRNA_disease_neg = {}
    shuffled_indices = np.random.permutation(len(disease_name))
    neg_ratio = 0.25
    # 负样本产生方式一
    for index in shuffled_indices:
        if len(miRNA_disease_neg) == int(len(miRNA_disease_pos) * neg_ratio):
            break
        neg_link = miRNA_name[index] + "::" + disease_name[-index]
        if neg_link in miRNA_disease_pos:
            continue
        if neg_link not in miRNA_disease_neg:
            miRNA_disease_neg[neg_link] = []
        miRNA_name_temp = miRNA_name[index]
        disease_name_temp = disease_name[-index]
        miRNA_feature = np.array(miRNA_vector[miRNA_name_temp])
        disease_feature = np.array(disease_vector[disease_name_temp])
        miRNA_disease_neg[neg_link] = feature_combination(miRNA_feature, disease_feature)
    """
    """
    # 负样本产生方式二
    import random
    miRNA_disease_neg = {}
    shuffled_indices = np.random.permutation(len(disease_name))
    neg_ratio = 0.2
    indices = np.random.choice(shuffled_indices, int(len(miRNA_disease_pos)*neg_ratio))
    while len(miRNA_disease_neg) < len(indices):
        ix_1 = random.choice(indices)
        ix_2 = random.choice(indices)
        if ix_1 == ix_2:
            continue
        neg_link = miRNA_name[ix_1] + "::" + disease_name[ix_2]
        if neg_link in miRNA_disease_pos:
            continue
        if neg_link not in miRNA_disease_neg:
            miRNA_disease_neg[neg_link] = []
        miRNA_name_temp = miRNA_name[ix_1]
        disease_name_temp = disease_name[ix_2]
        miRNA_feature = np.array(miRNA_vector[miRNA_name_temp])
        disease_feature = np.array(disease_vector[disease_name_temp])
        miRNA_disease_neg[neg_link] = feature_combination(miRNA_feature, disease_feature)
    """

    miRNA_disease_neg = get_neg_data(miRNA_disease_pos, method='2', neg_ratio=neg_ratio, feature_combination=feature_combination)
    # format data as dataframe
    miRNA_disease_pos_df = pd.DataFrame.from_dict(miRNA_disease_pos, orient='index')
    miRNA_disease_pos_df = miRNA_disease_pos_df.reset_index().rename(columns={'index': 'links'})
    miRNA_disease_pos_df['label'] = 1

    miRNA_disease_neg_df = pd.DataFrame.from_dict(miRNA_disease_neg, orient='index')
    miRNA_disease_neg_df = miRNA_disease_neg_df.reset_index().rename(columns={'index': 'links'})
    miRNA_disease_neg_df['label'] = 0

    return miRNA_disease_pos_df, miRNA_disease_neg_df


#######################################################################################################
#######################################################################################################
# 机器学习算法数据准备
# A.返回正负样本
def get_data(test_ratio=0.2, neg_ratio=0.2, feature_combination=feature_combination_Concatenate):
    miRNA_disease_pos_df, miRNA_disease_neg_df = get_pos_data(feature_combination=feature_combination, neg_ratio=neg_ratio)

    # miRNA_disease_data = pd.concat([miRNA_disease_pos_df, miRNA_disease_neg_df])
    miRNA_disease_data = miRNA_disease_pos_df.append(miRNA_disease_neg_df)
    miRNA_disease_data = miRNA_disease_data.reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(miRNA_disease_data, test_size=test_ratio, random_state=42)

    train_miRNA_disease_features = train_set.drop(["links", "label"], axis=1).copy()  # drop labels for train set
    train_miRNA_disease_labels = train_set["label"].copy()
    test_miRNA_disease_features = test_set.drop(["links", "label"], axis=1).copy()    # drop labels for train set
    test_miRNA_disease_labels = test_set["label"].copy()

    return train_miRNA_disease_features, train_miRNA_disease_labels, test_miRNA_disease_features, test_miRNA_disease_labels


X_train, y_train, X_test, y_test = get_data(test_ratio=0.3, neg_ratio=0.5,
                                            feature_combination=feature_combination_Concatenate)
"""
# a.feature_combination_Concatenate:
## w1000.l100.pp1.size32    64维
neg_ratio:  0.1     0.15    0.2     0.25    0.3     0.35    0.4     0.45    0.5     0.55    0.6     0.7     0.8
AUC:        1.0     0.998   1.0     1.0     0.999   1.0     0.997   0.996   0.997   0.995   1.0     1.0     0.997    
## w1000.l100.pp1.size64    128维
neg_ratio:  0.1     0.15    0.2     0.25    0.3     0.35    0.4     0.45    0.5     0.55    0.6     0.7     0.8
AUC:        0.972   0.939   0.655   0.907   0.936   0.931   0.963   0.975   0.955   0.974   0.981   0.966   0.973                                         
AUC(pp0):   0.647   0.802   0.755   0.739   0.741   0.698   0.725   0.782   0.752   0.786   0.738   0.783   0.746
                                 
# b.feature_combination_Hadamard:
## w1000.l100.pp1.size32    32维
neg_ratio:  0.1     0.15    0.2     0.25    0.3     0.35    0.4     0.45    0.5     0.55    0.6     0.7     0.8
AUC:        1.0     0.996   0.992   0.997   0.998   0.991   0.999   0.994   0.997   0.997   0.996   0.998   0.998                             
## w1000.l100.pp1.size64    64维
neg_ratio:  0.1     0.15    0.2     0.25    0.3     0.35    0.4     0.45    0.5     0.55    0.6     0.7     0.8
AUC:        0.901   0.843   0.677   0.807   0.775   0.772   0.768   0.895   0.858   0.735   0.884   0.863   0.762  
AUC(pp0):   0.638   0.646   0.772   0.766   0.585   0.726   0.755   0.752   0.762   0.645   0.633   0.816   0.736                                                                                                                          
"""
#######################################################################################################
#######################################################################################################
# B.机器学习算法——模型准备
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVR


def sgd_classifier(model=None, x_data=None, y_label=None, cv=10):
    print("*********************** Start {:}-Fold Cross Validation ***********************".format(cv))
    print("# Model: {:} Classifier".format(sys._getframe().f_code.co_name.split("_")[0].upper()))
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, x_data, y_label,
                             cv=cv, scoring="accuracy")
    for score, ix in zip(scores, range(1, cv+1)):
        print("## 第{:}-Fold Accuracy: {:}".format(ix, score))
    print("*********************** Cross Validation Done! ***********************\n")


sgd_clf = SGDClassifier(max_iter=5, tol=np.infty, random_state=42)
#sgd_classifier(sgd_clf, X_train, y_train, cv=10)


def svm_classifier(model=None, x_data=None, y_label=None, cv=10):
    model.fit(X_train, y_train)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, x_data, y_label,
                             scoring="neg_mean_squared_error", cv=cv)
    tree_rmse_scores = np.sqrt(-scores)
    print("交叉验证概况如下：")
    print(pd.Series(tree_rmse_scores).describe())


# svm_reg = SVR(kernel="rbf")
# svm_classifier(svm_reg, X_train, y_train, cv=10)

# C.Performance Measures
# 1.混淆矩阵
def get_confusion_matrix(y_train, y_train_pred):
    from sklearn.metrics import confusion_matrix
    print("*********************** Start Calculating Confusion Matrix ***********************")
    print("# 混淆矩阵: \n", confusion_matrix(y_train, y_train_pred))
    print("*********************** Calculating Confusion Matrix Done! ***********************\n")


# 2.Precision, Recall, F1 Score,and AUC
def get_score(y_train=None, y_train_pred=None):
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import roc_auc_score
    precision = precision_score(y_train, y_train_pred)
    recall = recall_score(y_train, y_train_pred)
    f1 = f1_score(y_train, y_train_pred)
    auc = roc_auc_score(y_train, y_train_pred)
    print("*********************** Start Calculating Scores ***********************")
    print("# Precision Score:\t{:}".format(precision))
    print("# Recall Score:\t{:}".format(recall))
    print("# F1 Score:\t{:}".format(f1))
    print("# AUC:\t{:}".format(auc))
    print("*********************** Calculating Scores Done! ***********************\n")
    return precision, recall, f1, auc



from sklearn.model_selection import cross_val_predict
"""
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=10)
get_confusion_matrix(y_train, y_train_pred)
get_score(y_train, y_train_pred)

sgd_classifier(sgd_clf, X_test, y_test, cv=10)
y_test_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=10)
get_confusion_matrix(y_test, y_test_pred)
get_score(y_test, y_test_pred)
"""
#######################################################################################################
#######################################################################################################
# 3.Plot Precision, Recall, ROC Curves
from sklearn.metrics import precision_recall_curve


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("PR vs threshold curve", fontsize=16)
    plt.ylim([0, 1])
    plt.show()


# The PR Curve
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls[:-1], precisions[:-1], "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("PR curve", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.figure(figsize=(8, 6))
    plt.show()


# The ROC Curve
# plots the TPR against FPR
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.show()


def plot_roc_curve_compare(x1, y1, x2=None, y2=None, label1=None, label2=None):
    plt.plot(x1, y1, 'r:', linewidth=2, label=label1)
    plt.plot(x2, y2, 'b-', linewidth=2, label=label2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower right", fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Comparing ROC curve', fontsize=16)
    plt.show()


# y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=10, method="decision_function")
# precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plot_precision_vs_recall(precisions, recalls)

"""
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
plot_roc_curve(fpr, tpr)


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=10, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, y_scores_forest)
plot_roc_curve_compare(fpr, tpr, fpr_forest, tpr_forest, "SGD", "Random Forest")
print("# AUC for Random Forest: ", roc_auc_score(y_train, y_scores_forest))

print("\n----------------------> start calculating scores for Random Forest")
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train, cv=10)
get_score(y_train, y_train_pred_forest)
"""
#######################################################################################################
#######################################################################################################
# Log result
def log_result_comparing(model=sgd_clf, test_ratio=0.2, neg_ratio_ls=None, feature_combination_ls=None):
    output_path_dir = "output_results/"
    vector_name = ".".join(VECTOR_PATH.split("/")[1].split(".")[2:6])
    outfilename = vector_name + ".result-comparing-for-different-negative-ratio.txt"
    outfile = open(output_path_dir + outfilename, 'w')
    for feature_combination in feature_combination_ls:
        output = "Method of Feature Combination: {}     Test Ratio: {}\n".format(str(feature_combination).split("_")[2].split()[0],
                                                                          test_ratio)
        output += "\t{:^20}\t{:^20}\t{:^20}\t{:^20}\t{:^20}\n".format('Negative Ratio', 'Precision|(Test)',
                                                                      'Recall|(Test)', 'F1|(Test)', 'AUC|(Test)')
        #output += "\tNegativeRatio\tPrecision|(Test)\tRecall|(Test)\tF1|(Test)\tAUC|(Test)\n"
        outfile.write(output)
        output = ""
        for neg_ratio in neg_ratio_ls:
            X_train, y_train, X_test, y_test = get_data(test_ratio=test_ratio, neg_ratio=neg_ratio,
                                                        feature_combination=feature_combination)
            y_train_pred = cross_val_predict(model, X_train, y_train, cv=10)
            get_confusion_matrix(y_train, y_train_pred)
            precision_score, recall_score, f1_score, auc_score = get_score(y_train=y_train, y_train_pred=y_train_pred)

            model_test = SGDClassifier(max_iter=5, tol=np.infty, random_state=42)
            model_test.fit(X_train, y_train)
            y_test_pred = model_test.predict(X_test)
            print("####################### Test Evaluation Start! #######################")
            get_confusion_matrix(y_test, y_test_pred)
            precision_score_test, recall_score_test, f1_score_test, auc_score_test = get_score(y_test, y_test_pred)
            print("####################### Test Evaluation Done! #######################\n")
            # write scores preparation, [:7] means store the first 5 digits
            neg_ratio_str = str(neg_ratio*100)[:5] + "%"
            precision_score_str = str(precision_score)[:7] + "|(" + str(precision_score_test)[:7] + ")"
            recall_score_str = str(recall_score)[:7] + "|(" + str(recall_score_test)[:7] + ")"
            f1_score_str = str(f1_score)[:7] + "|(" + str(f1_score_test)[:7] + ")"
            auc_score_str = str(auc_score)[:7] + "|(" + str(auc_score_test)[:7] + ")"
            output += "\t{:^20}\t{:^20}\t{:^20}\t{:^20}\t{:^20}\n".format(neg_ratio_str, precision_score_str,
                                                                          recall_score_str, f1_score_str, auc_score_str)
        outfile.write(output)
    outfile.close()


neg_ratio_ls = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95, 1.0,
                1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# neg_ratio_ls = [0.1, 0.15, 0.2]
feature_combination_ls = [feature_combination_Concatenate, feature_combination_Hadamard, feature_combination_Average]
log_result_comparing(model=sgd_clf, neg_ratio_ls=neg_ratio_ls, feature_combination_ls=feature_combination_ls)
#######################################################################################################
#######################################################################################################
"""
# 最初的代码
from sklearn.svm import SVR
svm_reg = SVR(kernel="rbf")
svm_reg.fit(train_miRNA_disease_features, train_miRNA_disease_labels)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(svm_reg, train_miRNA_disease_features, train_miRNA_disease_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("交叉验证概况如下：")
print(pd.Series(tree_rmse_scores).describe())

from sklearn.model_selection import cross_val_predict
y_scores = cross_val_predict(svm_reg, train_miRNA_disease_features, train_miRNA_disease_labels, cv=10)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_miRNA_disease_labels, y_scores)

import matplotlib.pyplot as plt
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
print('AUC: ', roc_auc_score(train_miRNA_disease_labels, y_scores))
"""

#######################################################################################################
#######################################################################################################
