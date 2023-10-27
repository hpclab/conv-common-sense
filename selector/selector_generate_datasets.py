#%% IMPORT NECESSARY LIBRARIES AND METHODS
import pandas as pd
import os
import sys
import pandas as pd
current = os.path.abspath('')
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(current)
from methods.generator_methods import * 
from methods.selector_methods import *


import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, help="Evaluation files path.",default='/data4/guidorocchietti/project_organized/data/trec/treccast/')
    parser.add_argument("--qrels_path",  type=str,  help="Qrels files path.",default='/data4/guidorocchietti/project_organized/data/trec/qrels/')
    parser.add_argument("--baseline_path",type =str, help='Baseline files path.', default='/data4/guidorocchietti/project_organized/sota/retrieval_results/baselines/perquery/')
    parser.add_argument("--data_path",type = str, help = 'Path of data with expansions and retrieval results', default = '/data4/guidorocchietti/project_organized/data/retrieval_results/new_knowledge/no_context/')
    parser.add_argument("--output_dir",type =str,help='Output save path',default='/data4/guidorocchietti/project_organized/data/results/new_knowledge/')
    args = parser.parse_args()
    print('Evaluation files path : ',args.eval_path)
    print('Qrels files path : ',args.qrels_path)
    print('Baseline files path : ',args.baseline_path)
    print('Path of data with expansions and retrieval results : ', args.data_path)
    print('Output save path : ', args.output_dir)
    print('Loading Evaluation, Qrels, Baselines and Data with metrics')


    ### LOAD EVALUATION, QRELS ,BASELINES AND DATA WITH METRICS
    all_evaluation = load_evaluation(args.eval_path).reset_index(drop=True)
    all_qrels = load_all_qrels(args.qrels_path).reset_index(drop=True)
    evaluation_for_qrels = all_evaluation[all_evaluation.qid.isin(all_qrels.qid.unique())]
    if any('2019' in file for file in os.listdir(args.baseline_path)):
        baselines = pd.DataFrame()
        for file in os.listdir(args.baseline_path):
            baselines = pd.concat([baselines, pd.read_csv(args.baseline_path+file, index_col =0 )])
    else:
        baselines = pd.read_csv(f'{args.baseline_path}baselines.csv',index_col=0)
    #dataset = {}
    if any('2019' in file for file in os.listdir(args.data_path)):
        data_with_metrics = pd.DataFrame()
        for file in os.listdir(args.data_path):
            if 'error' not in file:
                #dataset[file] = pd.read_csv(args.data_path+file, index_col = 0)
                data_with_metrics = pd.concat([data_with_metrics, pd.read_csv(args.data_path+file, index_col = 0)])#,names= [str(i) for i in range(40)])]) 
    else:
        data_with_metrics = pd.read_csv(f'{args.data_path}data_with_metrics.csv',index_col=0)
    #data_with_metrics = data_with_metrics[~(data_with_metrics['11'] == 'query')]
    data_with_metrics = data_with_metrics.dropna(axis =1,how ='all')
    ### JOIN THE DATA WITH METRICS WITH THE BASELINES
    df = data_with_metrics.merge(baselines[baselines['name'] == 'DPH'],how = 'inner', on='qid', suffixes=['','_raw'])
    #evaluation_for_qrels['context'] = generate_context(evaluation_for_qrels)
    df = df.merge(evaluation_for_qrels, on ='qid', suffixes =['','']).fillna('')

    ### GENERATE TRAINING FOR RECALL AND NDCG MARKING ALL THE EXPANSIONS THAT INCREASE THE CHOSEN METRIC AS POSITIVE
    ### AND THE WORST PERFORMING ONES AS NEGATIVE
    training_all_recall = generate_training(df,'recall_50')
    training_all_ndcg = generate_training(df,'ndcg_cut_1')

    ### CREATE TRAINING SETS BALANCED AND UNBALANCED AND SAVE THEM TO FILE
    balanced_training_recall = balance_df(training_all_recall)
    balanced_training_ndcg = balance_df(training_all_ndcg)
    unbalanced_training_recall = balance_df(training_all_recall,balance_value =4)
    unbalanced_training_ndcg = balance_df(training_all_ndcg,balance_value =4)
    if not(os.path.isdir(args.output_dir+'all/')): os.mkdir(args.output_dir+'all/')
    balanced_training_recall.to_csv(f'{args.output_dir}all/balanced_training_recall_50.csv')
    balanced_training_ndcg.to_csv(f'{args.output_dir}all/balanced_training_ndcg_cut_1.csv')
    unbalanced_training_recall.to_csv(f'{args.output_dir}all/unbalanced_training_recall_50.csv')
    unbalanced_training_ndcg.to_csv(f'{args.output_dir}all/unbalanced_training_ndcg_cut_1.csv')
    ### CREATE THE SLIPT FOR A K FOLD FOR EACH CONFIGURATION LISTED ABOVE
    if not(os.path.isdir(f'{args.output_dir}splits/')):os.mkdir(f'{args.output_dir}splits/')
    create_splits(balanced_training_recall, split_num = 4, output_path = f'{args.output_dir}splits/balanced_recall_50/')
    create_splits(balanced_training_ndcg, split_num = 4, output_path =  f'{args.output_dir}splits/balanced_ndcg_1/')
    create_splits(unbalanced_training_recall, split_num = 4, output_path =  f'{args.output_dir}splits/unbalanced_recall_50/')
    create_splits(unbalanced_training_ndcg, split_num = 4, output_path =  f'{args.output_dir}splits/unbalanced_ndcg_50/')


if __name__ == "__main__":
    main()


