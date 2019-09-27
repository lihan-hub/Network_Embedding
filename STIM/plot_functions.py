from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np


def log_jpg_path(k_fold=""):
    # date_now = datetime.now().strftime("%Y-%m-%d-%H-Precision_Recall_Roc_log/")
    date_now = datetime.now().strftime("%Y-%m-%d-Precision_Recall_Roc_log/")
    roc_path = "output_pictures/{:}/fold_{:}/roc/".format(date_now, k_fold)
    pre_path = "output_pictures/{:}/fold_{:}/precision/".format(date_now, k_fold)
    rec_path = "output_pictures/{:}/fold_{:}/recall/".format(date_now, k_fold)

    for path in [roc_path, pre_path, rec_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        """
        # remove all original files before saving files again
        else:
            files = os.listdir(path)
            for file in files:
                file_path = os.path.join(path, file)
                os.remove(file_path)
        """
    return roc_path, pre_path, rec_path


# bar plot for top-k
def top_k_precision(precision_ls, save_fig=False, disease_name=" ", file2save=""):
    top_k_ls = np.arange(10, 101, 10)
    bar_width = 5
    plt.bar(top_k_ls[:], precision_ls, width=bar_width, color='r', alpha=0.5)
    plt.xlabel("Top-k")
    plt.ylabel("Precision")
    plt.title(disease_name)
    plt.xticks(top_k_ls[:])
    plt.yticks(np.arange(0, 1.1, 0.2))
    # 添加数据标签
    for x, y in zip(top_k_ls[:], precision_ls):
        plt.text(x, y + 0.005, np.around(y, 2), ha='center', va='bottom', fontsize=10)
    if save_fig:
        plt.savefig(file2save, dpi=100)
        plt.show()
        plt.close()
    else:
        plt.show()


def top_k_recall(recall_ls, save_fig=False, disease_name=" ", file2save=""):
    top_k_ls = np.arange(10, 101, 10)
    bar_width = 5
    plt.bar(top_k_ls[:], recall_ls, width=bar_width, color='c', alpha=0.5)
    plt.xlabel("Top-k")
    plt.ylabel("Recall")
    plt.title(disease_name)
    plt.xticks(top_k_ls[:])
    plt.yticks(np.arange(0, 1.1, 0.2))
    # 添加数据标签
    for x, y in zip(top_k_ls[:], recall_ls):
        plt.text(x, y + 0.005, np.around(y, 2), ha='center', va='bottom', fontsize=10)
    if save_fig:
        plt.savefig(file2save, dpi=100)
        plt.show()
        plt.close()
    else:
        plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, save_fig=False, disease_name=" ", file2save=""):
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(disease_name)
    plt.legend(loc="lower right")
    if save_fig:
        plt.savefig(file2save, dpi=100)
        plt.show()
        plt.close()
    else:
        plt.show()
