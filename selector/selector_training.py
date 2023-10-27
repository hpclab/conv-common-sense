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

def train_for_splits(splits_path, training_type,label_columns = ['neg', 'pos'], output = '',lr = 0.0004,epochs = 8, find_lr = False, class_weight= None):
    if splits_path[-1] != '/': splits_path += '/'
    data_types = os.listdir(splits_path)
    output = output+training_type+'/'
    if not os.path.isdir(output):
        os.mkdir(output)  
    for type in data_types:
        files = os.listdir(splits_path+type+'/train/')
        newoutput = output + type +'/'
        if not os.path.isdir(newoutput):
            os.mkdir(newoutput)
        for file in files:
            df = pd.read_csv(splits_path+type+'/train/'+file).fillna('')
            df = df[['qid','conv_id' ,'turn','raw_utterance','expansion','pos','neg']]
            context_df = df[['qid','raw_utterance']].drop_duplicates()
            context_df['context'] =generate_context(context_df, n_turns=0)
            df = df.merge(context_df[['qid','context']], on='qid')
            last_path = newoutput + file.replace('.csv','')
            if not os.path.isdir(last_path):
                os.mkdir(last_path)
            train_classifier(df, training_type = training_type, label_columns = label_columns, output =last_path ,MODEL_NAME = 'bert-base-uncased' , batch_size =8, 
                        lr = lr,epochs =epochs, class_weight = class_weight , find_lr =find_lr, fit_one_cycle=False)
        
#%% LOAD EVALUATION, QRELS ,BASELINES AND DATA WITH METRICS
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_path", type=str, help="Evaluation files path.",default='/data4/guidorocchietti/project_organized/data/results/new_knowledge/splits/')
    parser.add_argument('--output_dir',type =str,help='Output save path',default='/data4/guidorocchietti/project_organized/selector/models/NDCG1_R50/no_context/')
    parser.add_argument('--epochs', type = int,default =30)
    parser.add_argument('--lr', type = bool,default = 10e-4)
    parser.add_argument('--find_lr', type = bool,default =False)
    args = parser.parse_args()
    training_types = ['CQ+E']#'C+QE','CQ+E']
    label_columns =['neg', 'pos']
    for training_type in training_types:
        train_for_splits(args.splits_path,training_type,output=args.output_dir,find_lr=args.find_lr,epochs=args.epochs)
    #print (os.listdir(args.splits_path))
    
    #train_for_splits(args.splits_path,output=args.output_dir)

# %% JOIN THE DATA WITH METRICS WITH THE BASELINES
#df = data_with_metrics.merge(baselines[[col for col in baselines.columns[:-1]]], on='qid', suffixes=['','_raw'])

#%%EXTRACT THE ORACLES WITH ALL THE BEST PERFORMING EXPANSIONS FOR RECALL AND NDCG AND SAVE THEM TO FILE
#recall_200_oracle, ndcg_cut_3_oracle = extract_oracle(df)
#recall_200_oracle.to_csv(f'{parent}/data/oracle/recall_with_all_best_expansions.csv')
#ndcg_cut_3_oracle.to_csv(f'{parent}/data/oracle/ndcg_with_all_best_expansions.csv')

#%%

if __name__ == "__main__":
    main()



#%%

