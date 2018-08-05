# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 14:27:39 2016

@author: Riivo
"""

import pandas as pd
import os


params = {"DATAPATH":"./issue_data/"}


def get_data_path():
    return params["DATAPATH"]

 
def load_data(path):
    def fix_columns(data):
        data.columns = map(lambda x: x.replace("`", ""), data.columns)
        #print data.columns
        if "created_at" in data.columns:
            if 'datetime64' not in str(data.created_at.dtype):
                data = data[data.created_at != "0000-00-00 00:00:00"].copy()
                #issues.created_at = pd.to_datetime(issues.created_at)            
            
            data["created_at"] =  pd.to_datetime(data["created_at"])
        if "actor_id" in data.columns:
            data = data[pd.notnull(data['actor_id'])]
        return data
    
    def _fc(cols):
        s = cols[:]
        try:
            s.remove("ext_ref_id")
        except:
            pass
        return s
    
    def filter_dates(issues1, key="created_at"):
        FILTER = pd.datetime(2011,6,1)
        FILTER2 = pd.datetime(2015,1,1)
        if key is None:
            return issues1[(issues1.index >= FILTER) & (issues1.index < FILTER2)]
        return issues1[(issues1[key] >= FILTER) & (issues1[key] < FILTER2) ]

    opts = {'escapechar':'\\', 'header':None,"na_values":'N',"parse_dates":["created_at"]}
    opts2 = {'escapechar':'\\', 'header':None,"na_values":'N'}
    
    com_head = ["project_id", "cid", "author_id", "committer_id", "c_project_id", "created_at"]   
    is_head = ["id","repo_id","reporter_id","assignee_id","pull_request","pull_request_id","created_at","ext_ref_id","issue_id"]
    ie_head = ["event_id","issue_id","actor_id","action","action_specific","created_at","ext_ref_id"]
    ic_head = ["issue_id","user_id","comment_id","created_at","ext_ref_id"]

    issues = pd.read_csv(os.path.join(path, "top100_all_isues_cat.csv"), names=is_head, usecols= _fc(is_head), **opts)
    issue_comments = pd.read_csv(os.path.join(path,"top100_all_isues_comments_cat.csv"), names=ic_head,usecols= _fc(ic_head),  **opts)
    issue_events = pd.read_csv(os.path.join(path, "top100_all_isues_events_cat.csv"), names=ie_head,usecols= _fc(ie_head),  **opts)
    project_commits = pd.read_csv(os.path.join(path, "top100_all_commits_cat.csv"),  names=com_head, usecols= _fc(com_head), **opts2) 
            

        
    issues = fix_columns(issues)
    issue_comments = fix_columns(issue_comments)
    issue_events = fix_columns(issue_events)
    project_commits = fix_columns(project_commits)
    
    issues = filter_dates(issues)
    issue_events = filter_dates(issue_events)
    issue_comments = filter_dates(issue_comments)
    project_commits = filter_dates(project_commits)
    
    
    return issues, issue_comments, issue_events,project_commits

