#%% IMPORT NECESSARY LIBRARIES AND METHODS
import pandas as pd
import os
import sys
import pandas as pd
current = os.path.abspath('')
parent = os.path.dirname(current)
sys.path.append(parent)
from methods.generator_methods import * 
from methods.selector_methods import *
import re
#%%

import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, help="Evaluation files path.",default='/data4/guidorocchietti/project_organized/data/trec/treccast/')
    parser.add_argument("--qrels_path",  type=str,  help="Qrels files path.",default='/data4/guidorocchietti/project_organized/data/trec/qrels/')
    parser.add_argument("--baseline_path",type =str, help='Baseline files path.', default='/data4/guidorocchietti/project_organized/data/trec/baselines/baselines.csv')
    parser.add_argument("--data_path",type = str, help = 'Path of data with expansions and retrieval results', default = '/data4/guidorocchietti/project_organized/data/results/new_knowledge/no_context/splits/')
    parser.add_argument("--models_path",type =str,help='folder where to find the trained models',default='/data4/guidorocchietti/project_organized/selector/models/NDCG1_R50/no_context/')
    
    args = parser.parse_args()
    all_evaluation = load_evaluation(args.eval_path).reset_index(drop=True)
    all_qrels = load_all_qrels(args.qrels_path).reset_index(drop=True)
    qrels19 = all_qrels[all_qrels.year.isin([2019,2020])]
    qrels21 = all_qrels[all_qrels.year.isin([2021])]

    evaluation_for_qrels = all_evaluation[all_evaluation.qid.isin(all_qrels.qid.unique())]
    baselines = pd.read_csv(args.baseline_path,index_col=0)
    #data_with_metrics_context = pd.read_csv(f'{parent}/data/trec/results/data_with_metrics.csv',index_col=0)
    datatypes = os.listdir(args.models_path)
    for data in datatypes:
        training_types = os.listdir(args.models_path+data+'/')
        for training in training_types:
            test, metrics = test_model(f'{args.models_path}{data}/{training}/',f'{args.data_path}{training}/')
            metrics.to_csv(f'{args.models_path}{data}/{training}/metrics.csv')
            test.to_csv(f'{args.models_path}{data}/{training}/metrics.csv')


if __name__ == '__main__':
    main()

'''
all_evaluation = load_evaluation('../data/trec/treccast/').reset_index(drop=True)
all_qrels = load_all_qrels('../data/trec/qrels/').reset_index(drop=True)
qrels19 = all_qrels[all_qrels.year.isin([2019,2020])]
qrels21 = all_qrels[all_qrels.year.isin([2021])]

evaluation_for_qrels = all_evaluation[all_evaluation.qid.isin(all_qrels.qid.unique())]
baselines = pd.read_csv(f'{parent}/data/trec/baselines/baselines.csv',index_col=0)
data_with_metrics_context = pd.read_csv(f'{parent}/data/trec/results/data_with_metrics.csv',index_col=0)
#%%

model_path= '/data4/guidorocchietti/project_organized/selector/models/NDCG1_R50/'
data_path = '/data4/guidorocchietti/project_organized/data/results/new_knowledge/splits/'
do_test = False
if do_test: 
    test_c_qe_balanced_ndcg, metrics_c_qe_balanced_ndcg = test_model(model_path+'C_QE/balanced_ndcg/',data_path+'balanced_ndcg/')
    test_c_qe_unbalanced_ndcg, metrics_c_qe_unbalanced_ndcg = test_model(model_path+'C_QE/unbalanced_ndcg/',data_path+'unbalanced_ndcg/')
    test_c_qe_balanced_recall, metrics_c_qe_balanced_recall = test_model(model_path+'C_QE/balanced_recall/',data_path+'balanced_recall/')
    test_c_qe_unbalanced_recall, metrics_c_qe_unbalanced_recall = test_model(model_path+'C_QE/unbalanced_recall/',data_path+'unbalanced_recall/')
    test_cq_e_balanced_ndcg, metrics_cq_e_balanced_ndcg = test_model(model_path+'CQ_E/balanced_ndcg/',data_path+'balanced_ndcg/')
    test_cq_e_unbalanced_ndcg, metrics_cq_e_unbalanced_ndcg = test_model(model_path+'CQ_E/unbalanced_ndcg/',data_path+'unbalanced_ndcg/')
    test_cq_e_balanced_recall, metrics_cq_e_balanced_recall = test_model(model_path+'CQ_E/balanced_recall/',data_path+'balanced_recall/')
    test_cq_e_unbalanced_recall, metrics_cq_e_unbalanced_recall = test_model(model_path+'CQ_E/unbalanced_recall/',data_path+'unbalanced_recall/')

    metrics_c_qe_balanced_ndcg.to_csv(model_path+'C_QE/balanced_ndcg/metrics.csv')
    metrics_c_qe_unbalanced_ndcg.to_csv(model_path+'C_QE/unbalanced_ndcg/metrics.csv')
    metrics_c_qe_balanced_recall.to_csv(model_path+'C_QE/balanced_recall/metrics.csv')
    metrics_c_qe_unbalanced_recall.to_csv(model_path+'C_QE/unbalanced_recall/metrics.csv')
    metrics_cq_e_balanced_ndcg.to_csv(model_path+'CQ_E/balanced_ndcg/metrics.csv')
    metrics_cq_e_unbalanced_ndcg.to_csv(model_path+'CQ_E/unbalanced_ndcg/metrics.csv')
    metrics_cq_e_balanced_recall.to_csv(model_path+'CQ_E/balanced_recall/metrics.csv')
    metrics_cq_e_unbalanced_recall.to_csv(model_path+'CQ_E/unbalanced_recall/metrics.csv')
else:
    metrics_c_qe_balanced_ndcg = pd.read_csv(model_path+'C_QE/balanced_ndcg/metrics.csv', index_col=0)
    metrics_c_qe_unbalanced_ndcg = pd.read_csv(model_path+'C_QE/unbalanced_ndcg/metrics.csv', index_col=0)
    metrics_c_qe_balanced_recall = pd.read_csv(model_path+'C_QE/balanced_recall/metrics.csv', index_col=0)
    metrics_c_qe_unbalanced_recall = pd.read_csv(model_path+'C_QE/unbalanced_recall/metrics.csv', index_col=0)
    metrics_cq_e_balanced_ndcg = pd.read_csv(model_path+'CQ_E/balanced_ndcg/metrics.csv', index_col=0)
    metrics_cq_e_unbalanced_ndcg = pd.read_csv(model_path+'CQ_E/unbalanced_ndcg/metrics.csv', index_col=0)
    metrics_cq_e_balanced_recall = pd.read_csv(model_path+'CQ_E/balanced_recall/metrics.csv', index_col=0)
    metrics_cq_e_unbalanced_recall = pd.read_csv(model_path+'CQ_E/unbalanced_recall/metrics.csv', index_col=0)
# %%

df_with_all_expansions = pd.read_csv('/data4/guidorocchietti/project_organized/data/results/data_with_metrics.csv',index_col =0).fillna('')
df = data_with_metrics.merge(baselines[[col for col in baselines.columns[:-1]]], on='qid', suffixes=['','_raw'])

do_test_prediction = False
if do_test_prediction:
    for training_type in ['C_QE','CQ_E']:
        for data_type in ['balanced_recall','unbalanced_recall','balanced_ndcg','unbalanced_ndcg']:
            path = f'{model_path}{training_type}/{data_type}/'
            df_copy = df.iloc[:]
            df_with_pred = test_predictions(path,df_copy)
            df_with_pred.to_csv(f'{path}df_with_pred.csv')

#%%
dt_path = '/data4/guidorocchietti/project_organized/selector/models/numberbatch/'
training_types  =  ['C_QE','CQ_E']
data_types = ['balanced_recall','unbalanced_recall','balanced_ndcg','unbalanced_ndcg']
name = 'df_with_pred.csv'
predictions = {}
summary = pd.DataFrame()
for training_type in training_types:
    for data_type in data_types:
        df = pd.read_csv(f'{dt_path}{training_type}/{data_type}/{name}', index_col = 0).fillna('')
        best_prediction = df.sort_values(['qid','positive_probability'], ascending = False).groupby('qid').first().reset_index()
        #prediction = extract_best_predictions(df)
        predictions[f'{training_type}_{data_type}'] = best_prediction
        summ = best_prediction.describe().iloc[1:2][['ndcg_cut_3','recall_200','ndcg_cut_3_raw','recall_200_raw']]
        summ.insert(0, 'name',f'{training_type}_{data_type}')
        summary = pd.concat([summary,summ])
#summary.to_csv(f'{current}/statistics/results_summary.csv')

# %%
if False:
    predictions_with_presence_summary_stemmed = pd.DataFrame()
    predictions_with_presence_summary_not_stemmed = pd.DataFrame()

    for key in predictions.keys():
        df_presence_stemmed = is_there_new_knowledge(predictions[key],do_stemming = True)
        df_presence_not_stemmed = is_there_new_knowledge(predictions[key],do_stemming = False)
        columns = [x for x in df_presence_stemmed.columns if x not in ['qid','conv_id','turn','year']]
        new_knowledge_contribution_stemmed = df_presence_stemmed[df_presence_stemmed.presence_value == 0].describe().iloc[1:2][['ndcg_cut_3','recall_200','ndcg_cut_3_raw','recall_200_raw' ]]
        old_knowledge_contribution_stemmed = df_presence_stemmed[df_presence_stemmed.presence_value == 1].describe().iloc[1:2][['ndcg_cut_3','recall_200','ndcg_cut_3_raw','recall_200_raw' ]]

        new_knowledge_contribution_non_stemmed = df_presence_not_stemmed[df_presence_not_stemmed.presence_value == 0].describe().iloc[1:2][['ndcg_cut_3','recall_200','ndcg_cut_3_raw','recall_200_raw' ]]
        old_knowledge_contribution_non_stemmed = df_presence_not_stemmed[df_presence_not_stemmed.presence_value == 1].describe().iloc[1:2][['ndcg_cut_3','recall_200','ndcg_cut_3_raw','recall_200_raw' ]]

        ndcg_diff_new = (new_knowledge_contribution_stemmed.ndcg_cut_3 - new_knowledge_contribution_stemmed.ndcg_cut_3_raw).values[0]
        recall_diff_new = (new_knowledge_contribution_stemmed.recall_200 - new_knowledge_contribution_stemmed.recall_200_raw).values[0]
        ndcg_diff_old = (old_knowledge_contribution_stemmed.ndcg_cut_3 - old_knowledge_contribution_stemmed.ndcg_cut_3_raw).values[0]
        recall_diff_old = (old_knowledge_contribution_stemmed.recall_200 - old_knowledge_contribution_stemmed.recall_200_raw).values[0]

        ndcg_diff_new_not_stemmed = (new_knowledge_contribution_non_stemmed.ndcg_cut_3 - new_knowledge_contribution_non_stemmed.ndcg_cut_3_raw).values[0]
        recall_diff_new_not_stemmed = (new_knowledge_contribution_non_stemmed.recall_200 - new_knowledge_contribution_non_stemmed.recall_200_raw).values[0]
        ndcg_diff_old_not_stemmed = (old_knowledge_contribution_non_stemmed.ndcg_cut_3 - old_knowledge_contribution_non_stemmed.ndcg_cut_3_raw).values[0]
        recall_diff_old_not_stemmed = (old_knowledge_contribution_non_stemmed.recall_200 - old_knowledge_contribution_non_stemmed.recall_200_raw).values[0]

        summary_stemmed = df_presence_stemmed[columns].describe().iloc[1:2]
        summary_stemmed.insert(0,'name',key)
        summary_stemmed.insert(len(summary_stemmed.columns),'ndcg_increase_new_knowledge',ndcg_diff_new)
        summary_stemmed.insert(len(summary_stemmed.columns),'recall_increase_new_knowledge',recall_diff_new)
        summary_stemmed.insert(len(summary_stemmed.columns),'ndcg_increase_context_knowledge',ndcg_diff_old)
        summary_stemmed.insert(len(summary_stemmed.columns),'recall_increase_context_knowledge',recall_diff_old)

        summary_not_stemmed = df_presence_not_stemmed[columns].describe().iloc[1:2]
        summary_not_stemmed.insert(0,'name',key)
        summary_not_stemmed.insert(len(summary_not_stemmed.columns),'ndcg_increase_new_knowledge',ndcg_diff_new_not_stemmed)
        summary_not_stemmed.insert(len(summary_not_stemmed.columns),'recall_increase_new_knowledge',recall_diff_new_not_stemmed)
        summary_not_stemmed.insert(len(summary_not_stemmed.columns),'ndcg_increase_context_knowledge',ndcg_diff_old_not_stemmed)
        summary_not_stemmed.insert(len(summary_not_stemmed.columns),'recall_increase_context_knowledge',recall_diff_old_not_stemmed)
        
        predictions_with_presence_summary_stemmed = pd.concat([predictions_with_presence_summary_stemmed, summary_stemmed])
        predictions_with_presence_summary_not_stemmed = pd.concat([predictions_with_presence_summary_not_stemmed, summary_not_stemmed])
    predictions_with_presence_summary_stemmed.to_excel(f'{current}/statistics/results_summary_with_presence_stemmed.xlsx')
    predictions_with_presence_summary_not_stemmed.to_excel(f'{current}/statistics/results_summary_with_presence_not_stemmed.xlsx')

    predictions_with_presence_summary_not_stemmed[['name' ,'presence_value',	'ndcg_increase_new_knowledge',	'recall_increase_new_knowledge', 'ndcg_increase_context_knowledge',	'recall_increase_context_knowledge']].round(3).to_csv(f'{current}/statistics/results_summary_with_presence_not_stemmed_tabular.txt', sep = '&')
    predictions_with_presence_summary_stemmed[['name','presence_value',	'ndcg_increase_new_knowledge',	'recall_increase_new_knowledge', 'ndcg_increase_context_knowledge',	'recall_increase_context_knowledge']].round(3).to_csv(f'{current}/statistics/results_summary_with_presence_stemmed_tabular.txt', sep = '&')

#%%
dt_path = '/data4/guidorocchietti/project_organized/selector/models/numberbatch/'
training_types  =  ['C_QE','CQ_E']
data_types = ['balanced_recall','unbalanced_recall','balanced_ndcg','unbalanced_ndcg']
name = 'df_with_pred.csv'
summary = pd.DataFrame()
for training_type in training_types:
    for data_type in data_types:
        df = pd.read_csv(f'{dt_path}{training_type}/{data_type}/{name}', index_col = 0).fillna('')
        best_prediction = df.sort_values(['qid','positive_probability'], ascending = False).groupby('qid').first().reset_index()
        bst19 = best_prediction[best_prediction.year.isin([2019,2020])]
        bst21 = best_prediction[best_prediction.year.isin([2021])]
        non_stemmed_results_19 = retrieve_ranked_results(bst19,qrels19,retrieval_methods = ['DPH'],text_column = 'query',eval_metrics = ['recall_200','ndcg_cut_3'],index_path = '../indexes/cast_2019_2020_non_stemmed', expand =False)
        non_stemmed_results_21 = retrieve_ranked_results(bst21,qrels21,retrieval_methods = ['DPH'],text_column = 'query',eval_metrics = ['recall_200','ndcg_cut_3'],index_path = '../indexes/cast_2021_non_stemmed', expand =False)
        non_stemmed_all = pd.concat([non_stemmed_results_19[0],non_stemmed_results_21[0]])
        non_stemmed_all.to_csv(f'/data4/guidorocchietti/project_organized/data/non_stemmed_results/{training_type}{data_types}.csv')
        #prediction = extract_best_predictions(df)
        
        summ = non_stemmed_all.describe().iloc[1:2]
        summ.insert(0, 'name',f'{training_type}_{data_type}')
        summary = pd.concat([summary,summ])
summary.to_csv(f'/data4/guidorocchietti/project_organized/data/non_stemmed_results/summary.csv')
# %%
'''