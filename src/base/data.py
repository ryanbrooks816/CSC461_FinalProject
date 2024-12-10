import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter
import random
import numpy as np
import pandas as pd

class LoopDataset(Dataset):
    def __init__(self, loop_ids, cfg):        
        self.EMB1 = [] # program embedding1 (reference)
        self.EMB2 = [] # program embedding2 (transformed)
        self.TR = [] # transformation
        self.Y = [] # speedups (target)
        self.loop_mapping = [] # mapping loop groups to their index in the dataset
        self.embed_dim = 0
        self.tr_dim = 0
        start_idx = 0        
        for l_id in loop_ids:
            fpath = os.path.join(cfg['data_path'], l_id, l_id)
            try:
                # load the embeddings (n_transfs x n_layers x emb_dim)
                embeds = getattr(self, f'load_{cfg["embedding_model"]}')(fpath)
            except:
                #print(f'Loop group {l_id} excluded: embedding not found')
                continue
            speedups = torch.load(f'{fpath}_speedup_refs.pt', weights_only=True, map_location='cpu')
            transforms = torch.load(f'{fpath}_transformation_encodings.pt', weights_only=True, map_location='cpu')
            n_transfs = speedups.shape[0]
            ref_fname = [f for f in os.listdir(os.path.join(cfg['data_path'],l_id)) if f.endswith('c.0.c')][0]
            ref_file_size = os.path.getsize(os.path.join(cfg['data_path'], l_id, ref_fname))
            assert(embeds.shape[0] == n_transfs == transforms.shape[0])
            if n_transfs < cfg['min_transformations'] or n_transfs > cfg['max_transformations']:
                #print(f'Loop {l_id} excluded: less than {cfg['min_transformations']} transformations')
                continue
            elif ref_file_size > cfg['max_source_size']:
                #print(f'Loop {l_id} excluded: file size over {cfg['max_source_size']} bytes')
                continue
            elif torch.max(speedups) > cfg['max_speedup']:
                # print(f'Loop {l_id} excluded: contains speedup over {cfg['max_speedup']} ({speedups})')
                continue
            else:
                if cfg['embedding_layer'] == 'last':
                    if cfg['n_embeddings'][0]:
                        self.EMB1.append(embeds[0,-1].repeat(n_transfs,1))
                    if cfg['n_embeddings'][1]:
                        self.EMB2.append(embeds[:,-1])
                if cfg['n_embeddings'][2]:
                    self.TR.append(transforms)
                self.Y.append(speedups)
                self.loop_mapping.append({'id':l_id, 'start':start_idx, 'end':start_idx+n_transfs})
                start_idx += n_transfs
        if cfg['n_embeddings'][0]:
            self.EMB1 = torch.vstack(self.EMB1).float()
            self.embed_dim = self.EMB1.shape[1]
        if cfg['n_embeddings'][1]:
            self.EMB2 = torch.vstack(self.EMB2).float()
            self.embed_dim = self.EMB2.shape[1]
        if cfg['n_embeddings'][2]:            
            self.TR = torch.vstack(self.TR).float()
            self.tr_dim += self.TR.shape[1]
        print(self.embed_dim)
        print(self.tr_dim)
        self.Y = torch.hstack(self.Y).float()
        if cfg['task'] == 'classification':
            self.Y = torch.bucketize(self.Y, torch.tensor(cfg['class_splits']), right=True) - 1
        print(f'{len(loop_ids)-len(self.loop_mapping)} loop groups removed for this split')
        # if cfg['n_embeddings'] == 2:
        #     assert self.EMB1.shape[0] == self.EMB2.shape[0] == self.TR.shape[0] == self.Y.shape[0]
        #     print(f'EMB1: {self.EMB1.shape} EMB2: {self.EMB2.shape} TR: {self.TR.shape} Y: {self.Y.shape}')
        # elif cfg['n_embeddings'] == 1:
        #     assert self.EMB1.shape[0] == self.TR.shape[0] == self.Y.shape[0]
        #     print(f'EMB1: {self.EMB1.shape} TR: {self.TR.shape} Y: {self.Y.shape}')

    def load_llvm_llmcompiler(self, fpath):
        #emb = torch.load(f'{fpath}_llvm_llmcompiler_7b_ftd_0.pt', weights_only=True, map_location='cpu')        
        return torch.load(f'{fpath}_llvm_llm_compiler_13b_all.pt', weights_only=True, map_location='cpu') 
    
    def load_source_llmcompiler(self, fpath):
        return torch.load(f'{fpath}_source_llmcompiler_13b_ftd_all.pt', weights_only=True, map_location='cpu')        

    def load_source_codellama(self, fpath):
        return torch.load(f'{fpath}_source_codellama_13b_hf_all.pt', weights_only=True, map_location='cpu')

    def load_source_codet5p(self, fpath):
        emb = torch.load(f'{fpath}_source_codet5p_110m_embedding_all.pt', weights_only=True, map_location='cpu')
        return emb.unsqueeze(1)

    def load_source_coderankembed(self, fpath):
        emb = torch.load(f'{fpath}_source_code_rank_embed_all.pt', weights_only=True, map_location='cpu')        
        return emb.unsqueeze(1)
        
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if self.EMB1 != []:
            if self.EMB2 != []:
                if self.TR != []:
                    return self.EMB1[idx], self.EMB2[idx], self.TR[idx], self.Y[idx]
                else:
                    return self.EMB1[idx], self.EMB2[idx], self.Y[idx]
            else:
                if self.TR != []:
                    return self.EMB1[idx], self.TR[idx], self.Y[idx]
                else:
                    return self.EMB1[idx], self.Y[idx]
        else:
            if self.EMB2 != []:
                if self.TR != []:
                    return self.EMB2[idx], self.TR[idx], self.Y[idx]
                else:
                    return self.EMB2[idx], self.Y[idx]
            else:
                if self.TR != []:
                    return self.TR[idx], self.Y[idx]
                else:
                    return self.Y[idx]
        
        # all_lists = [self.EMB1, self.EMB2, self.TR]
        # enabled_lists = [ls[idx] for ls in all_lists if ls != []]
        # print(enabled_lists)
        # return tuple(enabled_lists)

        # if self.TR != []:
        #     if self.EMB2 != []:
        #         return self.EMB1[idx], self.EMB2[idx], self.TR[idx], self.Y[idx]
        #     else:
        #         return self.EMB1[idx], self.TR[idx], self.Y[idx]
        # else:
        #     if self.EMB2 != []:
        #         return self.EMB1[idx], self.EMB2[idx], self.Y[idx]
        #     else:
        #         return self.EMB1[idx], self.Y[idx]

# returns three dataloaders for training, validation and testing
def get_data_loaders(cfg):
    # grab all available loop groups
    # loop_ids = [d for d in os.listdir(cfg['data_path']) if len(d) == 36]
    df = pd.read_csv(cfg['csv_path'])
    loop_ids = filter_loop_ids(df, cfg['filters'])
    # perform train/val/test split
    if cfg['stratification'] == 'random':
        tr_ids, va = train_test_split(loop_ids, test_size=0.2)
        va_ids, te_ids = train_test_split(va, test_size=0.5)
    else:
        if cfg['stratification'] == 'binary':
            loop_labels = balanced_loop_groups_binary(loop_ids, cfg['data_path'])
        elif cfg['stratification'] == 'clustered':
            loop_labels = balanced_loop_groups_clustered(loop_ids, cfg['data_path'])
        print(f'Loop groups stratified into {Counter(loop_labels)}')
        tr_ids, va, _, lab_va = train_test_split(loop_ids, loop_labels, test_size=0.2, stratify=loop_labels)
        va_ids, te_ids = train_test_split(va, test_size=0.5, stratify=lab_va)
    print(f'Total loop groups: {len(loop_ids)} Train: {len(tr_ids)} Val: {len(va_ids)} Test: {len(te_ids)}')
    tr_d = LoopDataset(tr_ids, cfg)
    va_d = LoopDataset(va_ids, cfg)
    te_d = LoopDataset(te_ids, cfg)
    print(f'Total datapoints: {len(tr_d)+len(va_d)+len(te_d)} Train: {len(tr_d)} Valid: {len(va_d)} Test: {len(te_d)}')
    return (
        DataLoader(tr_d, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['n_workers'], pin_memory=True),
        DataLoader(va_d, batch_size=len(va_d), shuffle=False, num_workers=cfg['n_workers'], pin_memory=True),
        DataLoader(te_d, batch_size=len(te_d), shuffle=False, num_workers=cfg['n_workers'], pin_memory=True)
    )

# create another data loader function that implements leave-one-out cross-validation
# returns a tuple of training and validation data loaders for each fold
def get_data_loaders_loocv(cfg):
    df = pd.read_csv(cfg['csv_path'])
    # group ids by collection
    applications = get_benchmark_applications(df, cfg['benchmark'])
    print(f'Applications: {applications}')
    loop_ids = {
        app: filter_loop_ids(df, cfg['filters'] + [('application', '==', app)])
        for app in applications
    }
        
    data_loaders = []
    for va_app in applications:
        print(f'Validation application: {va_app}')
        # validation set
        va_ids = loop_ids[va_app]
        va_d = LoopDataset(va_ids, cfg)
        # training set
        tr_ids = []
        for tr_app in applications:
            if tr_app != va_app:
                tr_ids += loop_ids[tr_app]
        tr_d = LoopDataset(tr_ids, cfg)
        print(f'Total datapoints: {len(tr_d)+len(va_d)} Train: {len(tr_d)} Valid: {len(va_d)}')
        data_loaders.append((
            va_app,
            DataLoader(tr_d, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['n_workers'], pin_memory=True),
            DataLoader(va_d, batch_size=len(va_d), shuffle=False, num_workers=cfg['n_workers'], pin_memory=True)
        ))
    return data_loaders

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# returns three dataloaders for training, validation and testing
def get_data_loaders_wandb(cfg):
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    # grab all available loop groups
    # loop_ids = [d for d in os.listdir(cfg['data_path']) if len(d) == 36]
    df = pd.read_csv(cfg['csv_path'])
    loop_ids = filter_loop_ids(df, cfg['filters'])
    # perform train/val/test split
    if cfg['stratification'] == 'random':
        tr_ids, va_ids = train_test_split(loop_ids, test_size=0.2, random_state=cfg['seed'])
    else:
        if cfg['stratification'] == 'binary':
            loop_labels = balanced_loop_groups_binary(loop_ids, cfg['data_path'])
        elif cfg['stratification'] == 'clustered':
            loop_labels = balanced_loop_groups_clustered(loop_ids, cfg['data_path'])
        elif cfg['stratification'] == 'majority':
            loop_labels = balanced_loop_groups_majority(loop_ids, cfg['data_path'], cfg['class_splits'], cfg['classes'])
        print(f'Loop groups stratified into {Counter(loop_labels)}')
        tr_ids, va_ids, _, lab_va = train_test_split(loop_ids, loop_labels, test_size=0.2, stratify=loop_labels)
    print(f'Total loop groups: {len(loop_ids)} Train: {len(tr_ids)} Val: {len(va_ids)}')
    tr_d = LoopDataset(tr_ids, cfg)
    va_d = LoopDataset(va_ids, cfg)
    print(f'Total datapoints: {len(tr_d)+len(va_d)} Train: {len(tr_d)} Valid: {len(va_d)}')
    # sampler = torch.utils.data.RandomSampler(tr_d, generator=torch.Generator().manual_seed(cfg['seed']))
    # set_seed(cfg['seed'])
    return (
        DataLoader(tr_d, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['n_workers'], pin_memory=True, worker_init_fn=seed_worker),
        DataLoader(va_d, batch_size=len(va_d), shuffle=False, num_workers=cfg['n_workers'], pin_memory=True)
    )

# returns T/F labels for each loop group based on 
# speedup values, true if loop group contains at least 
# one transformation with speedup >= 1, false otherwise
def balanced_loop_groups_binary(loop_ids, data_path):
    loop_id_labels = []
    for l_id in loop_ids:
        fname = os.path.join(data_path, l_id, f'{l_id}_speedup_refs.pt')
        speedups = torch.load(fname, weights_only=True, map_location='cpu')
        good_speedups = torch.where(speedups > 1, 1, 0).sum()
        loop_id_labels.append('good' if good_speedups > 0 else 'bad')
    return loop_id_labels

# returns cluster labels for each loop group based on
# the speedup distribution statistics
def balanced_loop_groups_clustered(loop_ids, data_path):
    loop_id_labels = []
    all_stats = []
    for l_id in loop_ids:
        fname = os.path.join(data_path, l_id, f'{l_id}_speedup_refs.pt')
        speedups = torch.load(fname, weights_only=True, map_location='cpu')
        stats = torch.tensor([speedups.max(), speedups.min(), speedups.quantile(0.25), speedups.quantile(0.75), torch.exp(torch.log(speedups).mean())])
        all_stats.append(stats)
    all_stats = torch.vstack(all_stats)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=4, n_init=50, init='k-means++'))
    ])
    loop_id_labels = [f'group_{l}' for l in pipe.fit_predict(all_stats)]
    return loop_id_labels

def balanced_loop_groups_majority(loop_ids, data_path, splits, classes):
    loop_id_labels = []
    for l_id in loop_ids:
        fname = os.path.join(data_path, l_id, f'{l_id}_speedup_refs.pt')
        speedups = torch.load(fname, weights_only=True, map_location='cpu')
        loop_id_stats = []
        for i in range(len(splits)-1):
            loop_id_stats.append(torch.where((speedups > splits[i]) & (speedups < splits[i+1]), 1, 0).sum())
        loop_id_labels.append(classes[loop_id_stats.index(max(loop_id_stats))])
    return loop_id_labels


# Get the loop group IDs that satisfy the given filters 
# filters: [(column, comparator, value),]
def filter_loop_ids(df, filters):
    # Apply each filter with the given comparator
    for column, comparator, value in filters:
        if comparator == '==':
            df = df[df[column] == value]
        elif comparator == '!=':
            df = df[df[column] != value]
        elif comparator == '>':
            df = df[df[column] > value]
        elif comparator == '<':
            df = df[df[column] < value]
        elif comparator == '>=':
            df = df[df[column] >= value]
        elif comparator == '<=':
            df = df[df[column] <= value]
        else:
            raise ValueError(f"Unsupported comparator: {comparator}")
    
    # Get the unique values from the 'id' column
    unique_ids = df['id'].unique().tolist()
    
    return unique_ids

def get_benchmark_applications(df, benchmark):
    return df[df['benchmark'] == benchmark]['application'].unique()
