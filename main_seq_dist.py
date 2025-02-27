import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pandas as pd
from utils_seq import *
from models_seq import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', type=str, default='Distribution',
                    choices=['Classification','Regression','Distribution'])
parser.add_argument('--dist_dim', type=int, default=301, help='Dimension of probability distribution.')
parser.add_argument('--seed', type=int, default=5, help='Random seed.')
parser.add_argument('--epochs', type=int, default=22,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
# parser.add_argument('--hidden', type=int, default=256,
                    # help='Number of hidden units.')
parser.add_argument('--src_vocab_size', type=int, default=21) # number of amino acids + 'Empty'
parser.add_argument('--src_len', type=int, default=10)
# parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--model', type=str, default='Transformer',choices=['RNN','LSTM','Bi-LSTM','Transformer'])
parser.add_argument('--model_path', type=str, default='model_val_best.pt')
parser.add_argument('--base_dir', type = str, default='/dfs9/tw/yuanmis1/mrsec/ML-MD-Peptide/DL_for_Peptide')

args = parser.parse_args()

# Transformer Parameters
if args.model == 'Transformer':
    args.d_model = 512  # Embedding size
    args.d_ff = 2048  # FeedForward dimension
    args.d_k = args.d_v = 64  # dimension of K(=Q), V
    args.n_layers = 6  # number of Encoder and Decoder Layer
    args.n_heads = 8  # number of heads in Multi-Head Attention

args.model_path=os.path.join(args.base_dir,'{}_lr_{}_bs_{}.pt'.format(args.model,args.lr,args.batch_size))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)  

def main():
    if args.task_type == 'Distribution':
        df_train = pd.read_csv('Sequential_Peptides/train_seqs_dist.csv')
        df_valid = pd.read_csv('Sequential_Peptides/valid_seqs_dist.csv')
        df_test = pd.read_csv('Sequential_Peptides/test_seqs_dist.csv')
        train_label = process_dist_labels(df_train, 'Label')
        valid_label = process_dist_labels(df_valid, 'Label').to(device)
        test_label = process_dist_labels(df_test, 'Label').to(device)
        assert train_label.shape[-1] == valid_label.shape[-1] == test_label.shape[-1] == args.dist_dim
    
    output_directory = 'results_seq_dist'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    args.max = train_label.max().item()
    args.min = train_label.min().item()
    train_feat = np.array(df_train["Feature"])
    valid_feat = np.array(df_valid["Feature"])
    test_feat = np.array(df_test["Feature"])

    train_enc_inputs = make_data(train_feat,args.src_len)
    valid_enc_inputs = make_data(valid_feat,args.src_len).to(device)
    test_enc_inputs = make_data(test_feat,args.src_len).to(device)

    train_loader = Data.DataLoader(MyDataSet(train_enc_inputs,train_label), args.batch_size, True)

    valid_mse_saved = 100
    valid_acc_saved = 0
    valid_kl_saved = 100
    loss_mse = torch.nn.MSELoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')

    if args.model == 'Transformer':
        model = Transformer(args).to(device)
        #optimizer = optim.SGD(model.parameters(), lr=0.2)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    for epoch in range(args.epochs):
        model.train()
        for enc_inputs,labels in train_loader:
            enc_inputs = enc_inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(enc_inputs)

            if args.task_type == "Distribution":
                loss = loss_kl(outputs, labels)

            # print('Epoch:','%04d' % (epoch+1), 'loss =','{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 1 == 0:
            model.eval()
            predict = model(valid_enc_inputs)
            print('valid_kl_saved',valid_kl_saved)
            if args.task_type == "Distribution":
                valid_kl = loss_kl(predict, valid_label)
                #valid_log_likelihood= (predict * valid_label).sum(dim=1).mean()
                if valid_kl_saved > valid_kl:
                    valid_kl_saved = valid_kl
                    print('Task Type:', args.task_type)
                    print('Epoch:',epoch+1)
                    print('Valid Performance:',epoch+1,valid_kl)
                    args.model_path_best=os.path.join(args.base_dir,'{}_lr_{}_bs_{}_best.pt'.format(args.model,args.lr,args.batch_size))
                    torch.save(model.state_dict(),args.model_path_best)#torch.save(model.state_dict(),args.model_path)
                    print(f"Model saved at: {args.model_path}")
    print('######## Final Epoch:',epoch+1)
    print('valid_kl_saved',valid_kl_saved)
    #torch.save(model.state_dict(), args.model_path)  # Save model explicitly
    print(f"Model saved at: {args.model_path_best}")

    
    predict = []

    if args.model == 'Transformer':
        model_load = Transformer(args).to(device)
    
    checkpoint = torch.load(args.model_path_best)
    model_load.load_state_dict(checkpoint)
    model_load.eval()
    
    outputs = model_load(test_enc_inputs)

    
    if args.task_type == 'Distribution':

        predict = outputs.exp().tolist()

        df_test_save = pd.DataFrame()
        labels = test_label.tolist()
        df_test_seq = pd.read_csv('Sequential_Peptides/test_seqs_dist.csv')
        df_test_save['feature'] = df_test_seq['Feature']
        df_test_save['predict'] = predict
        df_test_save['label'] = labels
        
        kl_divergences = F.kl_div(outputs, test_label, reduction='none') 
        #df_test_save['kl_divergences'] = kl_divergences.sum(dim=1).detach().numpy()
        df_test_save['kl_divergences'] = kl_divergences.sum(dim=1).detach().cpu().numpy()
        df_test_save.to_csv(os.path.join(output_directory,'Test_reg_{}_lr_{}_bs_{}.csv'.format(args.model,args.lr,args.batch_size)))
    #os.remove(args.model_path)

if __name__ == '__main__':
    main()
