# Reformat by xk_wang @ 2019.08.23
import deepwalk.graph as graph
from gensim.models import Word2Vec
import networkx as nx
from gcforest.gcforest import GCForest
import random
import math
from sklearn import metrics
import matplotlib
import pandas as pd
import tensorflow as tf
import xlrd
from plot_functions import *
from common_functions import *

np.random.seed(22)


def DDS_uniform(result_mat):
    """
    为了将原始矩阵DDS取值范围映射到0到1，同时让疾病节点之间越相似其距离计算值越大，
    即完全相似的疾病节点向量之间的距离度量值为1。
    对矩阵DDS进行归一化
    :param result_mat:      原始DDS矩阵
    :return:                对各行进行归一化后的矩阵
    """
    result = np.ones((result_mat.shape[0], result_mat.shape[1]))
    for i in range(result_mat.shape[0]):
        min_r = np.min(result_mat[i, :])
        max_r = np.max(result_mat[i, :])
        for j in range(result_mat.shape[1]):
            temp = (result_mat[i, j]-min_r)/(max_r-min_r)       # 距离越近，值越接近1
            result[i, j] = 1-temp
    return result


def dis_compute(deep, i, j):
    """
    :param deep:    向量矩阵
    :param i:       向量矩阵第i行
    :param j:       向量矩阵第j行
    :return:        向量距离相似度【DDS】
    """
    l1 = deep[i, :]
    l2 = deep[j, :]
    sum_value = 0.00
    for i in range(len(l1)):
        sum_value += ((l1[i] - l2[i]) * (l1[i] - l2[i]))
    if sum_value == 0:
        sum_value = 0
    else:
        sum_value = math.sqrt(sum_value)
    return sum_value


def compute(deep, i, j):
    """
    :param deep:    向量矩阵
    :param i:       向量矩阵第i行
    :param j:       向量矩阵第j行
    :return:        向量余弦相似度【CDS】
    """
    l1 = deep[i, :]
    l2 = deep[j, :]
    sum1 = 0.00
    sum2 = 0.00
    sum3 = 0.00
    for i in range(len(l1)):
        sum1 += (l1[i]*l2[i])
        sum2 += (l1[i]*l1[i])
        sum3 += (l2[i]*l2[i])
    return sum1/((math.sqrt(sum2))*(math.sqrt(sum3)))


def DBSI(disease_similaritity, data_train):
    """
    :param disease_similaritity:        disease相似度矩阵
    :param data_train:                  miRNA-disease邻接矩阵
    :return:                            疾病-miRNA预测分值——【相似的疾病可能存在相同的连接】
    """
    DBSI_result = np.ones((data_train.shape[1], data_train.shape[0]))*0
    for i in range(data_train.shape[1]):
        for j in range(data_train.shape[0]):
            val = 0.00
            sum_up = 0.00
            sum_low = 0.00
            for ix in range(data_train.shape[1]):
                if ix != i:
                    sum_up += (disease_similaritity[i, ix] * data_train[j, ix])
                    sum_low += (disease_similaritity[i, ix])
            val = sum_up/sum_low
            DBSI_result[i, j] = val
    return DBSI_result


def DBSI_uniform(result_mat):
    """
    :param result_mat:      原始疾病-miRNA预测分值矩阵
    :return:                按行归一化后的疾病-miRNA预测分值矩阵
    """
    for i in range(result_mat.shape[0]):
        max_r = np.max(result_mat[i, :])
        min_r = np.min(result_mat[i, :])
        for j in range(result_mat.shape[1]):
            low = max_r-min_r
            result_mat[i, j] = (result_mat[i, j]-min_r)/low
    return result_mat


def batch_generator(all_data, batch_size, shuffle=True):
    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    print("data_size: ", data_size)
    if shuffle:
        p = np.random.permutation(data_size)        # 返回打乱后的索引
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]


# 构建编码器
def encoder(x, n_input, n_hidden_1, n_hidden_2):
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),
    }
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# 构建解码器
def decoder(x, n_input, n_hidden_1, n_hidden_2):
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),
    }
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# 构建自编码器模型
def autocode(train_x, train_label, test_x, test_label):
    learning_rate = 0.01
    training_epochs = 10
    batch_size = 256
    display_step = 1
    n_input =train_x.shape[1]
    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, n_input])
    n_hidden_1 = 512  # 第一编码层神经元个数
    n_hidden_2 = 256  # 第二编码层神经元个数
    """
    n_hidden_1 = 256  # 第一编码层神经元个数
    n_hidden_2 = 128  # 第二编码层神经元个数
    """
    encoder_op = encoder(X, n_input, n_hidden_1, n_hidden_2)
    decoder_op = decoder(encoder_op, n_input, n_hidden_1, n_hidden_2)
    # 预测
    encode_x = encoder_op
    y_pred = decoder_op
    y_true = X
    # 定义代价函数和优化器
    cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))  # 最小二乘法
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
        # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
        train_x = np.array(train_x)
        train_label = np.array(train_label).reshape(-1, 1)
        batch_gen = batch_generator([train_x, train_label], batch_size)
        total_batch = int(len(train_x) / batch_size)  # 总批数
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_x, batch_y = next(batch_gen)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
        print("Optimization Finished!")
        result_train_x = sess.run(encode_x, feed_dict={X: train_x})
        result_test_x = sess.run(encode_x, feed_dict={X: test_x})
    return result_train_x, result_test_x


# 配置gcForest
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 0
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": 30, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 1, "type": "ExtraTreesClassifier", "n_estimators": 30, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 1, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


def getkolddata(data):
    print("\n生成五折交叉验证数据......")
    result = []
    lens = int(len(data)/5)
    print("前四折中，每折样本量:", lens)
    data_new = data.copy()
    for i in range(4):
        data = np.array(data_new)
        indices = np.random.choice(data, lens, replace=False)       # 不重复采样
        result.append(indices)
        data_new = []
        for val in data:                                            # 移除已采样的一折数据
            if val not in indices:
                data_new.append(val)
    print("第五折，样本量:", len(data_new))
    result.append(data_new)
    return result


def machine_def(disease_num, data, old_mRNA_similaritity, old_disease_similaritity, k_fold_data, index, str_name):
    key_len = 128                            # 嵌入向量维数
    # top_num = [50, 100, 150, 200, 250, 300]
    top_num = np.arange(10, 101, 10)
    disease_len = data.shape[1]
    mRNA_len = data.shape[0]
    auc_list = []

    index_train = []
    # index_train.append(random.randint(0, len(k_fold_data) - 1))     # 随机选取一折数据
    # for w in index_train:
    for k_fold in range(5):                 # 修改
        print("当前处理第{}折数据".format(k_fold+1))
        tf.reset_default_graph()
        indices = k_fold_data[k_fold]
        train_data = data.copy()
        has_assos = []
        for val in index:
            if val not in indices:
                has_assos.append(val)
        for val in indices:
            train_data[val, disease_num] = 0        # 屏蔽掉选取的一则数据

        a1 = data[:, disease_num]
        a1_one = np.ones(len(a1))
        a1_zero = a1_one - a1
        result_false = np.nonzero(a1_zero)[0]
        # **********************DeepWalk部分特征*********************************
        print("\n运行DeepWalk, 生成节点嵌入向量......")
        G = nx.Graph()
        node_counts = train_data.shape[1] + train_data.shape[0]
        for i in range(1, node_counts + 1):             # 注意：是从1开始编码的
            G.add_node(i)
        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[1]):
                if train_data[i, j] == 1:
                    G.add_edge(disease_len + i + 1, j + 1)
        nx.write_adjlist(G, "machine.adjlist")
        Gra = graph.load_adjacencylist("machine.adjlist")
        walks = graph.build_deepwalk_corpus(Gra, num_paths=100, path_length=50, alpha=0, rand=random.Random(0))
        # walks = graph.build_deepwalk_corpus(Gra, num_paths=40, path_length=40, alpha=0, rand=random.Random(0))
        model = Word2Vec(walks, size=key_len, window=5, min_count=0, sg=1, hs=1, workers=1)
        model.wv.save_word2vec_format("deep_machine.txt")

        print("读取生成的节点嵌入向量......")
        deepwork_data = open("deep_machine.txt")        # 读取嵌入向量
        fr = deepwork_data.readlines()
        deep_mRNA = np.ones((data.shape[0], key_len)) * 0
        deep_disease = np.ones((data.shape[1], key_len)) * 0
        for i in range(1, len(fr)):
            da = fr[i].split(" ")
            if int(da[0]) < (disease_len + 1):
                for j in range(1, len(da)):
                    deep_disease[int(da[0]) - 1, j - 1] = (float(da[j]))
            if int(da[0]) > disease_len:
                for j in range(1, len(da)):
                    deep_mRNA[int(da[0]) - disease_len - 1, j - 1] = (float(da[j]))
        # *************************获取训练和测试数据集*************************
        print("\n生成训练数据、测试数据......")
        tests_label = []
        tests_data = []
        train_true = []
        train_false = []
        really_train_false = []

        simlarility_tests_label = []
        simlarility_tests_data = []
        simlarility_train_true = []
        simlarility_train_false = []
        simlarility_really_train_false = []
        for i in range(train_data.shape[0]):
            for j in range(train_data.shape[1]):
                if j != disease_num:
                    da = []
                    for p in range(key_len):
                        da.append(deep_mRNA[i, p])
                    for q in range(key_len):
                        da.append(deep_disease[j, q])                   # 级联所有miRNA-disease嵌入向量，除了当前疾病
                    if train_data[i, j] == 1:
                        train_true.append(list(da))                     # 正样本
                    else:
                        train_false.append(list(da))                    # 负样本

                    da_sim = []
                    for p in range(mRNA_len):
                        da_sim.append(old_mRNA_similaritity[i, p])
                    for q in range(disease_len):
                        da_sim.append(old_disease_similaritity[j, q])   # 级联所有miRNA-disease相似度向量
                    if train_data[i, j] == 1:
                        simlarility_train_true.append(list(da_sim))     # 正样本
                    else:
                        simlarility_train_false.append(list(da_sim))    # 负样本
        print("Done! 级联所有miRNA-disease向量")
        for i in range(len(has_assos)):
            da = []
            for p in range(key_len):
                da.append(deep_mRNA[has_assos[i], p])
            for q in range(key_len):
                da.append(deep_disease[disease_num, q])
            train_true.append(list(da))                                 # 添加上当前疾病的少一折正样本，miRNA-disease嵌入向量

            da_sim = []
            for p in range(mRNA_len):
                da_sim.append(old_mRNA_similaritity[has_assos[i], p])
            for q in range(disease_len):
                da_sim.append(old_disease_similaritity[disease_num, q])
            simlarility_train_true.append(list(da_sim))                 # 添加上当前疾病的少一折正样本，miRNA-disease相似度向量
        print("当前第{}折训练数据, 正样本数: {}".format(k_fold + 1, len(train_true)))
        for i in range(len(indices)):
            da = []
            for p in range(key_len):
                da.append(deep_mRNA[indices[i], p])
            for q in range(key_len):
                da.append(deep_disease[disease_num, q])
            tests_data.append(list(da))
            tests_label.append(1)

            da_sim = []
            for p in range(mRNA_len):
                da_sim.append(old_mRNA_similaritity[indices[i], p])
            for q in range(disease_len):
                da_sim.append(old_disease_similaritity[disease_num, q])
            simlarility_tests_data.append(list(da_sim))
            simlarility_tests_label.append(1)
        print("当前第{}折测试数据, 正样本数: {}".format(k_fold + 1, len(tests_data)))

        for i in range(len(result_false)):
            da = []
            for p in range(key_len):
                da.append(deep_mRNA[result_false[i], p])        # result_false存储着当前disease的负样本的index
            for q in range(key_len):
                da.append(deep_disease[disease_num, q])
            tests_data.append(list(da))
            tests_label.append(0)

            da_sim = []
            for p in range(mRNA_len):
                da_sim.append(old_mRNA_similaritity[result_false[i], p])
            for q in range(disease_len):
                da_sim.append(old_disease_similaritity[disease_num, q])
            simlarility_tests_data.append(list(da_sim))
            simlarility_tests_label.append(0)
        # ***********************************嵌入向量方法***************************************
        # ***************************随机选择获取真正的训练数据集*******************************
        print("获取正样本:负样本 = 1:1的训练数据")
        get_trian = np.random.choice(a=len(train_false), size=len(train_true), replace=False)
        for num in get_trian:
            really_train_false.append(train_false[num])
        tf_len = len(train_true) + len(really_train_false)
        really_train_label = np.zeros(tf_len, int)
        for i in range(len(train_true)):
            really_train_label[i] = 1
        for i in range(len(really_train_false)):
            train_true.append(really_train_false[i])
        train_true = np.array(train_true)                       # 包含了所有正样本用于训练，除了一折剔除掉的正样本
        really_train_label = np.array(really_train_label)
        tests_data = np.array(tests_data)                       # 包含了一折剔除掉的正样本，用于测试
        tests_label = np.array(tests_label)
        # ***********************************相似向量方法***************************************
        # ***************************随机选择获取真正的训练数据集*******************************
        simlarility_get_trian = np.random.choice(a=len(simlarility_train_false), size=len(simlarility_train_true),
                                                 replace=False)
        for num in simlarility_get_trian:
            simlarility_really_train_false.append(simlarility_train_false[num])
        tf_len = len(simlarility_train_true) + len(simlarility_really_train_false)
        simlarility_really_train_label = np.zeros(tf_len, int)
        for i in range(len(simlarility_train_true)):
            simlarility_really_train_label[i] = 1
        for i in range(len(simlarility_really_train_false)):
            simlarility_train_true.append(simlarility_really_train_false[i])
        simlarility_train_true = np.array(simlarility_train_true)
        simlarility_really_train_label = np.array(simlarility_really_train_label)
        simlarility_tests_data = np.array(simlarility_tests_data)
        simlarility_tests_label = np.array(simlarility_tests_label)

        # *******************相似矩阵特征进行自编码器处理****************************
        print("\n相似矩阵特征——自编码器降维处理......")
        simlarility_train_true_code, simlarility_tests_data_code = autocode(simlarility_train_true,
                                                                            simlarility_really_train_label,
                                                                            simlarility_tests_data,
                                                                            simlarility_tests_label)
        # ************************进行模型训练和预测**********************
        print("开始训练模型......")
        config = get_toy_config()
        model = GCForest(config)
        model.set_keep_model_in_mem(True)
        model.fit_transform(np.array(train_true), np.array(really_train_label))

        model_simlarility = GCForest(config)
        model_simlarility.set_keep_model_in_mem(True)
        model_simlarility.fit_transform(np.array(simlarility_train_true_code),
                                        np.array(simlarility_really_train_label))
        print("\n开始预测......")
        weights_deep = model.predict_proba(np.array(tests_data))
        weights_simlarility = model_simlarility.predict_proba(np.array(simlarility_tests_data_code))

        r = 0.5
        weights = weights_deep * r + (1 - r) * weights_simlarility
        """
        print("开始训练模型......")
        # 级联相似性向量和嵌入向量
        train_emsemble_x = np.c_[normalization(np.array(train_true)), normalization(np.array(simlarility_train_true_code))]
        train_y_label = np.array(really_train_label)
        
        config = get_toy_config()
        model = GCForest(config)
        model.set_keep_model_in_mem(True)
        model.fit_transform(train_emsemble_x, train_y_label)
        print("\n开始预测......")
        test_emsemble_x = np.c_[normalization(np.array(tests_data)), normalization(np.array(simlarility_tests_data_code))]
        weights = model.predict_proba(test_emsemble_x)
        """
        # **********根据相似矩阵进行推断推荐*************************
        print("开始计算度量指标......")
        score1 = []
        score2 = []
        for i in range(len(weights)):
            score1.append(weights[i, 0])                # 预测为0的分值列表
            score2.append(weights[i, 1])                # 预测为1的分值列表
        top_result_pre = []
        score2 = np.array(score2)
        index_mRNA = np.argsort((-1) * score2)          # 按预测分值降序排列
        top_result_rec = []
        for i in range(len(top_num)):
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            val = top_num[i]
            for j in range(val):
                if index_mRNA[j] < len(indices):
                    TP += 1
                else:
                    FP += 1
            for j in range(val, len(weights)):
                if (index_mRNA[j]) < len(indices):
                    FN += 1
                else:
                    TN += 1
            pre = TP / val
            rec = TP / (TP + FN)
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            f1_score = 2 * (pre * rec) / (pre + rec)
            print("top-{}结果: ".format(val))
            print("Precision: {:.3f} ===> {:.2f}% ===> {:}/{:}".format(pre, pre * 100, TP, TP + FP))
            print("Recall: {:.3f} ===> {:.2f}% ===> {:}/{:}".format(rec, rec * 100, TP, TP + FN))
            print("Accuracy: {:.3f} ===> {:.2f}% ===> {:}/{:}".format(accuracy, accuracy*100, TP + TN, TP+TN+FP+FN))
            print("F1: {:.3f} ===> {:.2f}% ===> {:3f}/{:.3f}".format(f1_score, f1_score * 100, 2 * (pre * rec), pre + rec))
            top_result_pre.append(pre)
            top_result_rec.append(rec)

        pre_list = np.array(np.array(top_result_pre))       # 每折数据的top-k precision列表
        rec_list = np.array(np.array(top_result_rec))

        fpr, tpr, threshold = metrics.roc_curve(np.array(tests_label), score2)  # 计算真正率和假正率
        roc_auc = metrics.auc(fpr, tpr)

        print("AUC: ", np.around(roc_auc, 3))
        print("Top-k, Precision: ", np.around(pre_list, 3))
        print("Top-k, Recall: ", np.around(rec_list, 3))

        roc_path, pre_path, rec_path = log_jpg_path(k_fold=k_fold + 1)
        top_k_precision(precision_ls=pre_list, save_fig=True, disease_name=str_name, file2save=pre_path + str_name)
        top_k_recall(recall_ls=rec_list, save_fig=True, disease_name=str_name, file2save=rec_path + str_name)
        plot_roc_curve(fpr=fpr, tpr=tpr, roc_auc=roc_auc, save_fig=True, disease_name=str_name, file2save=roc_path+str_name)

        auc_list.append(np.around(roc_auc, 3))

    # 记录当前疾病的k-fold数据
    date_now = datetime.now().strftime("%Y-%m-%d-%H-AUC_log/")
    log_dir = "程序运行记录/" + date_now
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    f = open(log_dir + "log_auc{}.txt".format("【" + str_name + "】"), 'w')
    f.write("Disease\t#miRNA\tAUC #1st-fold\tAUC #2nd-fold\tAUC #3rd-fold\tAUC #4th-fold\tAUC #5th-fold\tAUC #Average\n")
    output = str_name + "\t" + str(len(index))
    sum_auc = 0.0
    for auc in auc_list:
        output += "\t" + str(auc) + "\t"
        sum_auc += auc
    aver_auc = sum_auc / len(auc_list)
    output += str(np.around(aver_auc, 3)) + "\n"
    f.write(output)
    f.close()

    return auc_list


def main_def(disease2clear_dict):
    disease_set3 = ['HepatocellularCarcinoma', 'SquamousCellCarcinoma', 'AcuteMyeloidLeukemia', 'RenalInsufficiency']
    disease_set1 = ['BreastNeoplasms', 'ColorectalNeoplasms']
    disease_set2 = ['Glioblastoma', 'HeartFailure', 'LungNeoplasms', 'Melanoma', 'OvarianNeoplasms',
                    'PancreaticNeoplasms', 'ProstaticNeoplasms', 'StomachNeoplasms', 'UrinaryBladderNeoplasms']
    disease_set = disease_set1 + disease_set2
    """
    这些疾病存在编码顺序问题：LeukemiaMyeloidAcute、CarcinomaHepatocellular、CarcinomaRenalCell、CarcinomaSquamousCell
    """
    # 需验证的15种疾病
    disease_one = ['LeukemiaMyeloidAcute', 'BreastNeoplasms', 'ColorectalNeoplasms', 'Glioblastoma', 'HeartFailure',
                   'CarcinomaHepatocellular', 'LungNeoplasms', 'Melanoma', 'OvarianNeoplasms', 'PancreaticNeoplasms',
                   'ProstaticNeoplasms', 'CarcinomaRenalCell', 'CarcinomaSquamousCell', 'StomachNeoplasms',
                   'UrinaryBladderNeoplasms']
    disease_name = []
    disease_dict = {}
    fr = open('disease_name.txt')
    for lines in fr.readlines():
        lines = lines.strip()
        disease_name.append(lines)
    i = 0
    for ke in disease_name:
        if ke in disease_dict.keys():
            print("erro")
        else:
            disease_dict[ke] = i
            i = i + 1
    print("disease 数量:", len(disease_dict.keys()))
    mRNA_name = []
    mRNA_dict = {}
    fr_m = open('mRNA_name.txt')
    for lines in fr_m.readlines():
        lines = lines.strip()
        mRNA_name.append(lines)
    i = 0
    for ke in mRNA_name:
        if ke in mRNA_dict.keys():
            print("erro")
        else:
            mRNA_dict[ke] = i
            i = i + 1
    print("miRNA 数量:", len(mRNA_dict.keys()))
    book = xlrd.open_workbook(filename=r"RD.xlsx")
    sheet = book.sheet_by_index(0)
    data = []
    for i in range(sheet.nrows):
        da = []
        for j in range(sheet.ncols):
            da.append(sheet.cell(i, j).value)
        data.append(da)
    data = np.array(data)
    print("miRNA-disease 邻接矩阵:", data.shape)
    # **********获取基于生物信息的RS和DS的矩阵****************************
    old_mRNA_similaritity = []
    old_disease_similaritity = []
    book_old_mRNA = xlrd.open_workbook(filename=r"RS.xlsx")
    sheet_old_mRNA = book_old_mRNA.sheet_by_index(0)
    for i in range(sheet_old_mRNA.nrows):
        da = []
        for j in range(sheet_old_mRNA.ncols):
            da.append(sheet_old_mRNA.cell(i, j).value)
        old_mRNA_similaritity.append(da)
    old_mRNA_similaritity = np.array(old_mRNA_similaritity)
    print("miRNA相似性矩阵:", old_mRNA_similaritity.shape)

    book_old_disease = xlrd.open_workbook(filename=r"DS.xlsx")
    sheet_old_disease = book_old_disease.sheet_by_index(0)
    for i in range(sheet_old_disease.nrows):
        da = []
        for j in range(sheet_old_disease.ncols):
            da.append(sheet_old_disease.cell(i, j).value)
        old_disease_similaritity.append(da)
    old_disease_similaritity = np.array(old_disease_similaritity)
    print("disease相似性矩阵:", old_disease_similaritity.shape)

    """
    for k_fold in range(5):
        auc_list = []
        print("当前处理第{}折数据".format(k_fold))
        date_now = datetime.now().strftime("%Y-%m-%d-%H-AUC_log/")
        log_dir = "程序运行记录/" + date_now
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
     """
    for str_name in disease_one[:]:
        disease_num = disease_dict[str_name]        # disease_id
        print("\n********************************************************************")
        print("********************************************************************")
        print("当前预测疾病: {}\tID: {}".format(disease2clear_dict[str_name], disease_num))
        disease = data[:, disease_num]
        disease.reshape(-1)
        index1 = np.nonzero(disease)
        index = np.array(index1)[0]                 # 返回与该疾病相连的miRNA_id列表【index】
        print("与当前疾病相关的所有miRNA index【ID】: 共{}个\n".format(len(index)), index)
        k_fold_data = getkolddata(index)
        # print("与当前疾病相关的k-fold数据:\n", k_fold_data)

        auc_list = machine_def(disease_num, data, old_mRNA_similaritity, old_disease_similaritity, k_fold_data, index, disease2clear_dict[str_name])


# 需验证的15种疾病
disease_one = ['LeukemiaMyeloidAcute', 'BreastNeoplasms', 'ColorectalNeoplasms', 'Glioblastoma', 'HeartFailure',
               'CarcinomaHepatocellular', 'LungNeoplasms', 'Melanoma', 'OvarianNeoplasms', 'PancreaticNeoplasms',
               'ProstaticNeoplasms', 'CarcinomaRenalCell', 'CarcinomaSquamousCell', 'StomachNeoplasms',
               'UrinaryBladderNeoplasms']
clear_disease_name = ['Acute myeloid leukemia', 'Breast neoplasms', 'Colorectal neoplasms', 'Glioblastoma',
                      'Heart failure', 'Hepatocellular carcinoma', 'Lung neoplasms', 'Melanoma',
                      'Ovarian neoplasms', 'Pancreatic neoplasms', 'Prostatic neoplasms', 'Renal cell carcinoma',
                      'Squamous cell carcinoma', 'Stomach neoplasms', 'Urinary bladder neoplasms']
disease2clear_dict = {}
for i in range(len(disease_one)):
    disease2clear_dict[disease_one[i]] = clear_disease_name[i]

main_def(disease2clear_dict)


