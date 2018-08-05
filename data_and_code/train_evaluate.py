# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:57:23 2015

@author: Riivo
"""
import os
import sys

N_THREADS = 4

import numpy as np
import pandas as pd
import itertools
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import chi2, SelectKBest

import matplotlib.pyplot as plt
import datetime
import seaborn as sns

import common_data as core_services

np.random.seed(10)

FEATURES_WHEN = 180
CLOSED_AT = 365

if len(sys.argv) >= 2:
    days = int(sys.argv[1])
    print "Calculating dynamic features after {0} days".format(days)
    sys.stdout.flush()
    FEATURES_WHEN = days

if len(sys.argv) >= 3:
    days = int(sys.argv[2])
    print "Predicting issue closure at {0} days".format(days)
    sys.stdout.flush()
    CLOSED_AT = days

experiment = None

if len(sys.argv) >= 4:
    experiment = sys.argv[3]

MULTICLASS = False
if CLOSED_AT == 366:
    MULTICLASS = True

# @cache_decorator.cache
BASEBATH= "./issue_data/"
DATAPATH1 = BASEBATH+""
MONGOPATH = BASEBATH+"text/"

FIGPATHATH1 = "figures/"

issues_orig = pd.read_csv(DATAPATH1 + "fixed_issues.csv")
repos = pd.read_csv(DATAPATH1 + "fixed_repos.csv")

issues_orig["created_at"] = pd.to_datetime(issues_orig["created_at"])
issues_orig["closed_at"] = pd.to_datetime(issues_orig["closed_at"])

issue_ds = pd.read_csv(DATAPATH1 + "feature_tables_days_{0}.csv".format(FEATURES_WHEN), index_col=0)

def fix_columns(cols):
    mapping = {
        "commits_by_commiters_before3m": "nCommitsByActorsT",
        "commits_by_commenters_nunique": "nCommitsByUniqueActorsT",
        "created_before3m": "nIssuesByCreator",
        "closed_before3m": "nIssuesByCreatorClosed",
        "commit_before3m": "nCommitsByCreator",
        "commit_before3m_nunique": "nCommitsByCreatorProjects",
        "comment_before_12month": "nCommentsT",
        "actors": "nActorsT",
        "ev_assigned": "nAssignmentsT",
        "ev_demilestoned": "nDemilestoningT",
        "ev_labeled": "nLabelsT",
        "ev_mentioned": "nMentionedByT",
        "ev_milestoned": "nMilestonedByT",
        "ev_referenced": "nReferencedByT",
        "ev_renamed": "nRenamedT",
        "ev_subscribed": "nSubscribedByT",
        "ev_unassigned": "nUnassingedByT",
        "ev_unlabeled": "nUnlabeledT",
        "body_len": "issueBodyLen",
        "title_len": "issueTitleLen",
        "comment_len_mean": "meanCommentSizeT",
        "comment_len_sum": "sumCommentSizeT",
        "out_mentions": "nPersonsMentionedBody",
        "out_sha1": "nCommitsMentionedBody",
        "out_issues": "nIssuesMentionedBody",
        "body_len_strip": "issueCleanedBodyLen",
        "codes": "nCodeBlocksInContent",
        "text_score": "textScore",
        "created_before_project3m": "nIssuesCreatedInProject",
        "closed_before_project3m": "nIssuesCreatedInProjectClosed",
        "commit_before_project3m": "nCommitsInProject",
        "commit_before_project3m_nunique": "nPersonCommitingInProject",
        "created_before_project2w_sliding": "nIssuesCreatedProjectT",
        "closed_before_project2w_sliding": "nIssuesCreatedProjectClosedT",
        "commit_before_project2w_sliding": "nCommitsProjectT",
        "commit_before_project2w_nunique_sliding": "nCommitsProjectUniqueT",
        "text_score_comments": "textScoreComments"
    }
    return [mapping[c] for c in cols]


def get_comments(FEATURES_WHEN):
    comments_text = pd.read_csv(MONGOPATH + "cleaned_text_comments.csv").rename(columns={"issue_id": "comment_id"})

    comments_text.loc[pd.isnull(comments_text.cleaned_text), "cleaned_text"] = ""
    issues, comments, event, _ = core_services.load_data(core_services.get_data_path())
    merged = pd.merge(comments[["issue_id", "comment_id", "created_at"]].copy(), comments_text, on="comment_id",
                      how="inner")

    renamed_issues = issues[["id", "created_at"]].copy().rename(columns={"id": "issue_id", "created_at": "origin_at"})
    merged = pd.merge(merged, renamed_issues, on="issue_id", how="inner")

    year_after = merged.origin_at.values + np.timedelta64(FEATURES_WHEN, 'D')
    merged = merged[merged.created_at.values <= year_after]

    merged.sort_values(["issue_id", "created_at"], inplace=True)
    grouped = merged[["issue_id", "cleaned_text"]].groupby("issue_id").apply(lambda x: " ".join(x["cleaned_text"]))

    return grouped.reset_index().rename(columns={0: "comment_text"})




final_mentions = pd.read_csv(MONGOPATH + "final_mentions.csv")
cleaned_text = pd.read_csv(MONGOPATH + "cleaned_text.csv")

comments_text = get_comments(FEATURES_WHEN)
cleaned_text.loc[pd.isnull(cleaned_text.cleaned_text), "cleaned_text"] = ""

missing_ids = set(issue_ds.issue_id.tolist()) - set(final_mentions.issue_id.tolist())

issue_ds = pd.merge(issue_ds, final_mentions, how="inner")
issue_ds = pd.merge(issue_ds, cleaned_text, how="inner")
print issue_ds.shape
issue_ds = pd.merge(issue_ds, comments_text, how="left")

issue_ds.loc[pd.isnull(issue_ds.comment_text), "comment_text"] = ""

print issue_ds.shape

cols = issue_ds.columns.values.tolist()
to_remove = ["issue_id", "elapsed", "closedin12", "issticky", "rid", "relative_month", "live_at_dynamic"]
to_remove.extend(['total_comments', 'comment_before_closure', 'uniform_comments', "cleaned_text", "comment_text"])
to_remove.extend(["label_count"])

unimportant = ["commit_before3m_nunique", "comment_len_sum", "body_len", "title_len", "ev_renamed", "ev_unlabeled"]
unimportant.extend(["out_issues", "ev_milestoned", "out_sha1", "ev_unassigned", "ev_demilestoned", "codes",
                    "commit_before_project3m_nunique"])
unimportant.extend(["out_mentions", "commit_before_project2w_nunique_sliding"])

to_remove.extend(unimportant)

to_remove.extend(filter(lambda x: x.find("lbl_") != -1, cols))
for x in to_remove:
    if x in cols:
        cols.remove(x)


# fix train/test splöit
subset = issues_orig[["issue_id", "created_at", "closed_at"]].copy()

too_short = 13
subset.loc[:, "train"] = too_short

endp = np.datetime64("2015-01-01")
end_point = endp.astype("datetime64[s]")

dt_test_start = np.datetime64("2013-09-01")
dt_train_point = dt_test_start.astype("datetime64[s]")

rows_not_closed = pd.isnull(subset.closed_at)

# training  created_before split
# test- created after split
# too short_train - non_closed created_at least x days before dt_tain_point
# too short_test - non_closed created_at least x days before end_point

# training live at least  x days before the split point
created_bef_split = (subset.loc[:, "created_at"].values) < dt_train_point
subset.loc[created_bef_split, "train"] = 1
subset.loc[~created_bef_split, "train"] = 0

# closed but not in our dataset
# train_over_to_test = (subset.loc[:,"closed_at"].values >=  dt_train_point)
# train_over_rows =  created_bef_split & train_over_to_test & (~rows_not_closed)
# subset.loc[train_over_rows, "train"] =  15


train_short = (subset.loc[:, "created_at"].values + np.timedelta64(CLOSED_AT, 'D')) > dt_train_point
train_short_rows = (subset.loc[:, "train"] == 1) & train_short & rows_not_closed
subset.loc[train_short_rows, "train"] = 13

test_short = (subset.loc[:, "created_at"].values + np.timedelta64(CLOSED_AT, 'D')) > end_point
test_short_rows = (subset.loc[:, "train"] == 0) & test_short & rows_not_closed
subset.loc[test_short_rows, "train"] = 14

print subset[["created_at", "closed_at", "train"]].groupby(["train"]).agg(["min", "max", "count"])



predicate = "all"
issue_ds = pd.merge(issue_ds[(issue_ds.live_at_dynamic == 1)], subset, on="issue_id")


issue_ds.loc[:, "closedin12"] = 0
rows = ~pd.isnull(issue_ds.closed_at)
year_after_created = issue_ds[rows].created_at.values + np.timedelta64(CLOSED_AT, 'D')
issue_ds.loc[rows, "closedin12"] = (issue_ds[rows].closed_at.values <= year_after_created) + 0


#  if it is overflown, then it must have been open for long in enough to observ
# hence it does not matter
train_over_to_test = (issue_ds.loc[:, "closed_at"].values >= dt_train_point) & (
        issue_ds.loc[:, "created_at"].values < dt_train_point)
issue_ds.loc[(train_over_to_test & rows), "closedin12"] = 0
elapsed = (issue_ds[rows].closed_at - issue_ds[rows].created_at).astype("timedelta64[s]") / 86400.0


print issue_ds[["created_at", "closed_at", "closedin12", "train"]].groupby(["closedin12", "train"]).agg(
    ["min", "max", "count"])

#
for a, b in itertools.combinations(cols, 2):

    val = stats.spearmanr(issue_ds[a].values, issue_ds[b].values).correlation
    if val > 0.5:
        print "%{0},{1},{2:.3f}".format(a, b, val)
    # print a, b, stats.spearmanr(issue_ds[a].values, issue_ds[b].values)

# %%
selector = SelectKBest(chi2, k=10)
ytrain = issue_ds[issue_ds.train == 1].closedin12.values
Xtrain = issue_ds[issue_ds.train == 1][cols].values
selector.fit(Xtrain, ytrain)

vals1 = zip(cols, selector.scores_)
for x, v in sorted(vals1, key=lambda y: y[1], reverse=True):
    print v, x



# %%
def bucket(val):
    b = [0, 1, 7, 14, 30, 90, 180, 365, 1200]
    # labels = ["D","W" "2W","M", "3M","HY","Y","MY"]
    cur = None
    for x, y in zip(b, b[1:]):
        if val >= x:
            cur = y
    return cur


issue_ds.loc[:, "multiclass"] = 1200
issue_ds.loc[rows, "multiclass"] = elapsed.map(lambda x: bucket(x))


ytrain = issue_ds[issue_ds.train == 1].closedin12.values
Xtrain = issue_ds[issue_ds.train == 1][cols].values

ytest = issue_ds[issue_ds.train == 0].closedin12.values
Xtest = issue_ds[issue_ds.train == 0][cols].values

if MULTICLASS:
    ytrain = issue_ds[issue_ds.train == 1].multiclass.values
    ytest = issue_ds[issue_ds.train == 0].multiclass.values



def add_text_score(Xtrain, Xtest, issue_ds, col="cleaned_text", col_new="text_score", cols=None):
    print "Text label", col, col_new
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 22, ngram_range=(1, 3), non_negative=True)
    # vectorizer = TfidfVectorizer(decode_error='ignore', max_features=5000,max_df=0.8, min_df=100, ngram_range=(1,2))

    Xtrain2 = vectorizer.fit_transform(issue_ds[issue_ds.train == 1][col].tolist())
    Xtest2 = vectorizer.transform(issue_ds[issue_ds.train == 0][col].tolist())

    scores = np.zeros((ytrain.shape[0], 1), dtype=np.float64)
    sgdkw = {"loss": "log", "verbose": 0, "penalty": 'l2', "alpha": 1e-3, "n_iter": 5, "random_state": 42,
             "class_weight": "balanced"}
    kfold = cross_validation.KFold(Xtrain.shape[0], 2)

    for train_index, test_index in kfold:
        bff = issue_ds[issue_ds.train == 1][col].values
        Xa = bff[train_index].tolist()
        Xb = bff[test_index].tolist()
        ya = ytrain[train_index]
        clf = SGDClassifier(**sgdkw)
        clf.fit(vectorizer.transform(Xa), ya)
        ybhat = clf.predict_proba(vectorizer.transform(Xb))
        scores[test_index, 0] = ybhat[:, 1]

    cols.append(col_new)
    Xtrain = np.append(Xtrain, scores, axis=1)
    # %%
    clf = SGDClassifier(**sgdkw)

    clf.fit(Xtrain2, ytrain)
    pred2 = clf.predict_proba(Xtest2)
    scores2 = np.zeros((ytest.shape[0], 1), dtype=np.float64)
    scores2[:, 0] = pred2[:, 1]
    Xtest = np.append(Xtest, scores2, axis=1)
    return Xtrain, Xtest, cols


Xtrain, Xtest, cols = add_text_score(Xtrain, Xtest, issue_ds, "cleaned_text", "text_score", cols=cols)
Xtrain, Xtest, cols = add_text_score(Xtrain, Xtest, issue_ds, "comment_text", "text_score_comments", cols=cols)


print "Down sample"
print "balance before", ytrain.sum() / float(ytrain.shape[0])

ntrain = ytrain.shape[0] * 1.0
pos_labels = ytrain.sum()
neg_labels = ntrain - ytrain.sum()

rate = pos_labels / ntrain
if rate < 0.35:
    print "f"
    ideal = 0.35
    reduce1 = ideal / rate
    final_size = ntrain / reduce1 - pos_labels
    if final_size >= 0.35 * pos_labels:
        yx = np.where(ytrain == 1)[0]
        yy = np.where(ytrain == 0)[0]
        yyp = np.random.choice(yy, int(final_size), replace=False)
        sel = np.hstack((yyp, yx))
        Xtrain = Xtrain[sel, :]
        ytrain = ytrain[sel]
        # Xtrain2 = Xtrain2[sel,:]

        print "subsampling done, pos class"

if rate > 0.65:
    print "f"
    ideal = 0.35
    reduce1 = ideal / (1.0 - rate)
    final_size = ntrain / reduce1 - neg_labels
    if final_size >= neg_labels:
        yx = np.where(ytrain == 1)[0]
        yy = np.where(ytrain == 0)[0]

        yyp = np.random.choice(yx, int(final_size), replace=False)
        sel = np.hstack((yyp, yy))
        Xtrain = Xtrain[sel, :]
        ytrain = ytrain[sel]
        # Xtrain2 = Xtrain2[sel,:]

        print "subsampling done, neg class"

print "balance after", ytrain.sum() / float(ytrain.shape[0])


clf = RandomForestClassifier(n_estimators=1000, n_jobs=N_THREADS, verbose=0, max_depth=5, random_state=1,
                             min_samples_leaf=1, class_weight="balanced")
clf.fit(Xtrain, ytrain)


method = "randomforest_ntrees=1000_m_depth=5"

print "Training done"
sys.stdout.flush()
ypred = clf.predict_proba(Xtest)

auc = -1.0
acc = None
f1 = None

if not MULTICLASS:
    auc = metrics.roc_auc_score(ytest, ypred[:, 1])
ypred_lbl = clf.predict(Xtest)

acc = metrics.accuracy_score(ytest, ypred_lbl)
f1 = metrics.f1_score(ytest, ypred_lbl, average="binary")

conf = metrics.confusion_matrix(ytest, ypred_lbl)
print conf

prec1 = metrics.precision_score(ytest, ypred_lbl, average="binary")
rec1 = metrics.recall_score(ytest, ypred_lbl, average="binary")
identifier = datetime.date.today().isoformat()

print "Obtained auc {0:.3f} Feautres at {1} Predction at {2}, ACC={3:.3f} F1={4:.3f}".format(auc, FEATURES_WHEN,
                                                                                             CLOSED_AT, acc, f1)
log_str = "{0:.3f},{1},{2},{3},{4},{5}, {6},{7},{8},{9},{10}, {11}, {12}, {13}\n".format(auc, FEATURES_WHEN, CLOSED_AT,
                                                                                         predicate,
                                                                                         datetime.datetime.now().isoformat(),
                                                                                         method, acc, f1,
                                                                                         ytrain.shape[0],
                                                                                         ytest.shape[0],
                                                                                         (ytrain.sum()) / (
                                                                                         1.0 * ytrain.shape[0]),
                                                                                         (ytest.sum()) / (
                                                                                         1.0 * ytest.shape[0]), prec1,
                                                                                         rec1)

print log_str
if experiment is None:
    fp = open("results-%s.csv" % (identifier), "a")
else:
    fp = open("results-%s.csv" % (experiment), "a")
fp.write(log_str)
fp.close()


test_out = pd.DataFrame(
    {"predlabel%d" % (CLOSED_AT): ypred_lbl, "predscore%d" % (CLOSED_AT): ypred[:, 1], "actual": ytest,
     "issue_id": issue_ds[issue_ds.train == 0].issue_id.values})

test_out.to_csv(FIGPATHATH1 + "test_output_%d_%d_predicate_%s.csv" % (FEATURES_WHEN, CLOSED_AT, predicate), index=False)

sns.set_style("whitegrid")
fig_size = [5, 4]

sns.set_context("paper", font_scale=0.9, rc={'figure.figsize': [3.1, 2.6]})
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure()
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(fix_columns(cols))[sorted_idx])
plt.ylim([0, len(pos)])
plt.xlabel('Relative Importance')

plt.tight_layout(0.2)

plt.savefig(FIGPATHATH1 + "feautre_importance_%d_%d_predicate_%s.pdf" % (FEATURES_WHEN, CLOSED_AT, predicate))
plt.close()


fp = open(FIGPATHATH1 + "feautre_importance_%d_%d_predicate_%s.csv" % (FEATURES_WHEN, CLOSED_AT, predicate), "w")
for f, c in zip(feature_importance, fix_columns(cols)):
    fp.write("{0:.4f}, {1}\n".format(f, c))

fp.flush()
fp.close()



fpr, tpr, _ = metrics.roc_curve(ytest, ypred[:, 1])
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(4, 2))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.tight_layout(0.2)
plt.savefig(FIGPATHATH1 + "roc_curve_%d_%d_predicate_%s.pdf" % (FEATURES_WHEN, CLOSED_AT, predicate))
