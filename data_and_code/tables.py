# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:40:59 2016

@author: Riivo
"""

import pandas as pd
from sklearn import metrics

DATAPATH1="./figures/"
RESULTS_FILE="results-all.csv"

def formatter(x):
    if pd.isnull(x):
        return ""
    return "%.3f" % x


def print_tables():
    df = pd.read_csv(RESULTS_FILE,delimiter=",")

    print "F1"
    table =  pd.pivot_table(df," F1"," FEATURES"," PREDICTION")
    print table.to_latex(float_format = formatter ,na_rep="")

    print "AUC"
    table =  pd.pivot_table(df,"AUC"," FEATURES"," PREDICTION")
    print table.to_latex(float_format = formatter ,na_rep="")

    print "ACC"
    table =  pd.pivot_table(df," ACC"," FEATURES"," PREDICTION")
    print table.to_latex(float_format = formatter ,na_rep="")


def get_str(ypred, y):
    labels = [1,0]
    cm = metrics.confusion_matrix(ypred,  y, labels)
    
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    
    return "{0} & {1} & {2} & {3}".format(TP,FP,FN,TN)


def stable_test_peformance():
    cnt=0

    for feat_point in [0,1,7,14,30,90,180,365]:
        for prediction_point in  [1,7,14,30,90,180,365]:
            if prediction_point > feat_point:
                cnt+=1

    points = [0,1,7,14,30,90,180,365]            

    template="test_output_{0}_{1}_predicate_all.csv"

    fp = open("results_fixedtest.csv", "w")
    for ij, prediction_point in enumerate(reversed(points[2:])):
       
        colls = []    
        un = None
        pointset = list(reversed(list(reversed(points))[ij+1:]))
        for j, feat_point in enumerate(pointset):
            fname = template.format(feat_point, prediction_point)
            lbl = "pred_score%d"%feat_point        
            plbl = "pred_label%d"%feat_point        
            rules = {"predscore%d"%(prediction_point):lbl, "predlabel%d"%(prediction_point):plbl}
            
            ds = pd.read_csv(DATAPATH1+fname).rename(columns=rules)
            colls.append(ds)
        
            if un is None:
                un = colls[j][["issue_id",lbl, "actual", plbl]].copy()
            else:
                un = pd.merge(un, colls[j][["issue_id",lbl, plbl]].copy(), on="issue_id")

        #print "\\midrule \n  & \\multicolumn{4}{c}{\\textbf{Closedin in %d days (N=%d)}} \\\\  \n\\midrule " % (prediction_point, un.shape[0])    
        fp.write("\\midrule \n  & \\multicolumn{8}{c}{\\textbf{Closedin in %d days (N=%d)}} \\\\  \n\\midrule \n" % (prediction_point, un.shape[0]))  
        
        for _, feat_point in  enumerate(pointset):
            lbl = "pred_score%d"%feat_point
            auc = metrics.roc_auc_score(un.actual.values, un[lbl].values)
            
            plbl = "pred_label%d"%feat_point 
            
            prec = metrics.precision_score(un.actual.values, un[plbl].values,average="binary")
            recall = metrics.recall_score(un.actual.values, un[plbl].values,average="binary")
            f1 = metrics.f1_score(un.actual.values, un[plbl].values, average="binary")
            #print feat_point        #print metrics.confusion_matrix(un[plbl].values,un.actual.values, labels=[1,0])
            trp =  get_str(un[plbl].values, un.actual.values)
            
            fp.write("%d & %.3f & %.3f &  %.3f & %.3f  & %s \\\\ \n" %(feat_point,auc, prec, recall, f1, trp))
            #print "%d & %.3f & %.3f &  %.3f & %s \\\\" %(feat_point,auc, prec, recall,get_str(un[plbl].values, un.actual.values))
    fp.close()

if __name__ == "__main__":
    stable_test_peformance()
    print_tables()