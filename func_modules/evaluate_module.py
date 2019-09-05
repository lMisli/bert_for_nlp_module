
"""BERT finetuning runner."""

# coding:utf-8
import matplotlib.pyplot as plt
import re
import logging
import os
import pandas as pd
from scripts import dataprocess, common
import numpy as np
import argparse

parser = argparse.ArgumentParser("run_evaluation")
parser.add_argument("--input_data", type=str, help="predict data path")
parser.add_argument("--output_dir", type=str, help="test results directory")
parser.add_argument("--pr", type=bool, default=True, help="Whether to calculate precision-recall(pr)")
parser.add_argument("--roc", type=bool, default=True, help="Whether to calculate receiver operating characteristic(ROC)")
parser.add_argument("--label_columns", type=str, help="[Optional]label columns,the order should responding to probability_columns. Separated by ' ' , such as 'col1 col2' default='[L|l]abel*'")
parser.add_argument("--probability_columns", type=str, help="[Optional]probability columns, the order should responding to label columns. Separated by ' ' , such as 'col1 col2', if None, default='[P|p]robability*'")

args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)


def evaluate():

    args.input_data = common.parse_path(args.input_data)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pred = dataprocess.load_data(args.input_data)


    if args.label_columns == None or args.label_columns == "" or args.label_columns == "null":
        label_columns = [x for x in pred.columns if re.match('[L|l]abel*', x) != None]
    else:
        label_columns = args.label_columns.split(' ')
    if args.probability_columns == None or args.probability_columns == "" or args.label_columns == "null":
        probability_columns = [x for x in pred.columns if re.match('[P|p]robability*', x) != None]
    else:
        probability_columns = args.probability_columns.split(' ')
    logging.info('num of labels: %d' %(len(label_columns)))
    logging.info('num of probability_columns: %d' %(len(probability_columns)))
    assert len(label_columns) == len(probability_columns)

    precision_vec = []
    recall_vec = []
    tpr_vec = []
    fpr_vec = []
    for i in range(len(label_columns)):
        logging.info('evaluating label: %s' % (label_columns[i]))

        drop_cols = label_columns + probability_columns
        drop_cols.remove(label_columns[i])
        drop_cols.remove(probability_columns[i])

        newpred1 = pred.sort_values(by=[probability_columns[i]], ascending=False, inplace=False)
        label_arr = newpred1[label_columns[i]]
        #preds_arr = newpred1[probability_columns[i]]
        pos_num = np.sum(label_arr == 1)
        neg_num = np.sum(label_arr == 0)
        if args.pr:
            logging.info('calculating precision-recall...')
            pre = []
            rec = []
            output_pr_file = os.path.join(args.output_dir, label_columns[i]+"_pr.csv")
            for j in range(len(label_arr)):
                if j == 0:
                    pre.append(1)
                    rec.append(0)
                else:
                    tp = np.sum((label_arr[:j] == 1))
                    pre.append(tp / j)
                    rec.append(tp / pos_num)

            pr_out = newpred1.drop(columns=drop_cols)
            pr_out['recall'] = rec
            pr_out['precision'] = pre
            logging.info('saving pricsion-recall results for label %s to %s' % (label_columns[i], output_pr_file))
            pr_out.to_csv(output_pr_file, index=0)

            precision_vec.append(pre)
            recall_vec.append(rec)
        if args.roc:
            logging.info('calculating receiver operating characteristic...')
            tpr = []
            fpr = []
            output_roc_file = os.path.join(args.output_dir, label_columns[i]+"_roc.csv")
            for j in range(len(label_arr)):
                tpr.append(np.sum(label_arr[:j] == 1) / pos_num)
                fpr.append(np.sum(label_arr[:j] == 0) / neg_num)
            roc_out = newpred1.drop(columns=drop_cols)
            roc_out['fpr'] = fpr
            roc_out['tpr'] = tpr
            logging.info('saving receiver operating characteristic results for label %s to %s' % (label_columns[i], output_roc_file))
            roc_out.to_csv(output_roc_file, index=0)
            tpr_vec.append(tpr)
            fpr_vec.append(fpr)
    precision_df = pd.DataFrame(np.array(precision_vec).T, columns=label_columns)
    recall_df = pd.DataFrame(np.array(recall_vec).T, columns=label_columns)
    tpr_df = pd.DataFrame(np.array(tpr_vec).T, columns=label_columns)
    fpr_df = pd.DataFrame(np.array(fpr_vec).T, columns=label_columns)

    return precision_df, recall_df, tpr_df, fpr_df




if __name__ == "__main__":
    # logging.info('hello')
    # args.output_dir = "output_dir_part"
    # args.input_data = "D:/corpus/toxic comment/jigsaw-toxic-comment-classification-challenge/predict_results.csv"
    # args.input_data = "D:/corpus/toxic comment/jigsaw-toxic-comment-classification-challenge/predict_results.csv"
    label_columns = "toxic,severe_toxic,obscene,threat,insult,identity_hate"
    p_df, r_df, tpr_df, fpr_df = evaluate()

    labels = p_df.columns.values.tolist()
    if (p_df.shape[0] != 0):
        assert p_df.shape == r_df.shape
        plt.figure(1)
        plt.title("Precision-Recall")
        for label in labels:
            plt.plot(r_df[label], p_df[label], label=label)
        plt.legend(loc='lower left')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(args.output_dir,'p-r.png'))
        #plt.show()
    if (tpr_df.shape[0] != 0):
        plt.figure(2)
        assert tpr_df.shape == fpr_df.shape
        plt.title('ROC')
        for label in labels:
            plt.plot(fpr_df[label], tpr_df[label], label=label)
        plt.legend(loc='lower right')
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.savefig(os.path.join(args.output_dir,'roc.png'))
        #plt.show()



