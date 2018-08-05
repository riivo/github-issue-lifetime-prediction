import os

exp_name = "all"
fp = open("results-%s.csv"%(exp_name), "w")
fp.write("AUC, FEATURES, PREDICTION, PREDICATE, WHEN, METHOD, ACC, F1, TRAIN_SIZE, TEST_SIZE, TRAIN_BALANCE, TEST_BALANCE, PRECISION, RECALL\n")
fp.close()

#%%
cnt=0
for feat_point in [0,1,7,14,30,90,180,365]:
    for prediction_point in  [1,7,14,30,90,180,365]:
        if prediction_point > feat_point:
            cnt+=1       
            os.system("python train_evaluate.py {0} {1} {2}".format(feat_point, prediction_point, exp_name))
            
print cnt

os.system("python tables.py")

df = {}
