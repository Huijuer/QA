import os
import sys

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from pruning_dataloader import DatasetPruning, DataLoaderPruning
from pruning_model import PruningModel
from torch.optim.lr_scheduler import ExponentialLR
import networkx as nx
from collections import defaultdict


parser = argparse.ArgumentParser()

parser.add_argument('--ls', type=float, default=0)
parser.add_argument('--validate_every', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--decay', type=float, default=0.5)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--nb_epochs', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--patience', type=int, default=20)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

args = parser.parse_args()


from multiprocessing import freeze_support
freeze_support()


def printRelationText(rel_ids, idx2rel):
    rel_text = idx2rel[rel_ids]
    print(rel_text)


def validate_v2(model, device, train_dataset, rel2idx, idx2rel):
    model.eval()
    data = process_data_file('my_qanoentity_train.txt', rel2idx, idx2rel)
    num_correct = 0
    count = 0
    correct = []
    for i in tqdm(range(len(data))):
        # try:
        d = data[i]
        question = d[0]
        question_tokenized, attention_mask = train_dataset.tokenize_question(question)
        question_tokenized = question_tokenized.to(device)
        attention_mask = attention_mask.to(device)
        rel_id = d[1]
        scores = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
        top2 = torch.topk(scores, 1)
        top2 = top2[1]
        if i < 5:
            # print("scores",scores)
            # print("question",question)
            # print("top2",top2)
            # print("question_tokenized",question_tokenized)
            # print("attention_mask",attention_mask)
            # print("rel_id",rel_id)
            print("-------------------")
        else:
            exit()
        int_top2 = top2.int()
        int_value = int_top2[0].item()

        if int_value == rel_id:
            num_correct += 1
        correct.append(str(int_value))
        # else:
        #     printRelationText(top2, idx2rel)
        #     printRelationText(rel_id, idx2rel)
        #     count += 1
        #     if count == 10:
        #         exit(0)
        # pred_rel_id = torch.argmax(scores).item()
        # if pred_rel_id == rel_id:
        #     num_correct += 1

    # np.save("scores_webqsp_complex.npy", scores_list)
    # exit(0)
    print(correct)
    accuracy = num_correct / len(data)
    return accuracy


def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return


def process_data_file(fname, rel2idx, idx2rel):
    f = open(fname, 'r')
    data = []
    for line in f:
        line = line.strip().split('\t')
        question = line[0].strip()
        # TODO only work for webqsp. to remove entity from metaqa, use something else
        # remove entity from question
        question1 = question.split('[')[0]
        # print(question1,question,line[0].strip())
        # sys.exit()
        rel = line[1]
        rel_id = rel2idx[rel]
        data.append((question, rel_id))
    return data


def train(batch_size, shuffle, num_workers, nb_epochs, gpu, use_cuda, patience, validate_every, lr, decay, ls):
    # f = open('/scratche/home/apoorv/mod_TuckER/models/ComplEx_fbwq_full/relations.dict', 'r')
    f = open(
        '../../data/my_data/relations.dict',
        'r')
    rel2idx = {}
    idx2rel = {}
    for line in f:
        line = line.strip().split('\t')
        id = int(line[1])
        rel = line[0]
        rel2idx[rel] = id
        idx2rel[id] = rel
    print(idx2rel)
    f.close()
    data = process_data_file('my_qanoentity_train.txt', rel2idx, idx2rel)
    device = torch.device(gpu if use_cuda else "cpu")
    # print(device)
    dataset = DatasetPruning(data=data, rel2idx=rel2idx, idx2rel=idx2rel)
    # print(dataset.data)
    # sys.exit()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = PruningModel(rel2idx, idx2rel, ls)
    # checkpoint_file = "checkpoints/pruning/best_best.pt"
    # checkpoint = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint)
    # print('loaded from ', checkpoint_file)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    for epoch in range(nb_epochs):
        phases = []
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    if i_batch >5:
                        exit()
                    model.zero_grad()
                    question_tokenized = a[0].to(device)
                    attention_mask = a[1].to(device)
                    rel_one_hot = a[2].to(device)
                    loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask,
                                 rel_one_hot=rel_one_hot)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss / ((i_batch + 1) * batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()

                scheduler.step()

            elif phase == 'valid':
                model.eval()
                eps = 0.0001
                score = validate_v2(model=model, device=device, train_dataset=dataset, rel2idx=rel2idx, idx2rel=idx2rel)
                if score > best_score + eps:
                    best_score = score
                    print("Validation accuracy up ", score)
                    file_path = "checkpoints/pruning/best_mar2_23.pt"
                    if os.path.exists(file_path):
                        print("Warning: File already exists. Overwriting existing file.")
                    else:
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    torch.save(model.state_dict(), file_path)
                elif score < best_score + eps:
                    print("Validation accuracy down ", score)
                #     no_update = 0
                #     best_model = model.state_dict()
                #     print("Validation accuracy increased from previous epoch", score)
                #     # writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
                #     file_path = "checkpoints/pruning/best_mar2_23.pt"
                #     if os.path.exists(file_path):
                #         print("Warning: File already exists. Overwriting existing file.")
                #     else:
                #         os.makedirs(os.path.dirname(file_path), exist_ok=True)
                #     torch.save(model.state_dict(), file_path)
                # elif (score < best_score + eps) and (no_update < patience):
                #     no_update += 1
                #     print("Validation accuracy decreases to %f from %f, %d more epoch to check" % (
                #     score, best_score, patience - no_update))
                # elif no_update == patience:
                #     print("Model has exceed patience. Saving best model and exiting")
                #     exit()
                # if epoch == nb_epochs - 1:
                #     print("Final Epoch has reached. Stoping and saving model.")
                #     exit()


def data_generator(data, roberta_file, entity2idx):
    question_embeddings = np.load(roberta_file, allow_pickle=True)
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1]
        # encoded_question = question_embedding[question]
        encoded_question = question_embeddings.item().get(question)
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question), ans, data_sample[1]


def Test(ls):
    f = open(
        '../../data/my_data/relations.dict',
        'r')
    rel2idx = {}
    idx2rel = {}
    for line in f:
        line = line.strip().split('\t')
        id = int(line[1])
        rel = line[0]
        rel2idx[rel] = id
        idx2rel[id] = rel
    f.close()
    f = open('../../data/QA_data/my_data/my_qatrain.txt', 'r')
    line = f.readline().strip().split('\t')
    line = f.readline().strip().split('\t')
    line = f.readline().strip().split('\t')
    question = line[0].strip()
    f.close()
    rel = line[1]
    rel_id = rel2idx[rel]
    data=(question, rel_id)
    device = torch.device(0)
    model = PruningModel(rel2idx, idx2rel, ls)
    model.load_state_dict(torch.load('checkpoints/pruning/best_mar2_23.pt'))
    model.to(device)
    model.eval()
    question = data[0]
    dataset = DatasetPruning(data=data, rel2idx=rel2idx, idx2rel=idx2rel)
    question_tokenized, attention_mask = dataset.tokenize_question(question)
    question_tokenized = question_tokenized.to(device)
    attention_mask = attention_mask.to(device)
    rel_id = data[1]
    scores = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
    top2 = torch.topk(scores, 1)
    top2 = top2[1]
    print("实际",rel_id,"预测",top2[0],top2)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    # Test(ls=args.ls)
    # sys.exit()
    train(
    batch_size=args.batch_size,
    shuffle=args.shuffle_data,
    num_workers=args.num_workers,
    nb_epochs=args.nb_epochs,
    gpu=args.gpu,
    use_cuda=args.use_cuda,
    patience=args.patience,
    validate_every=args.validate_every,
    lr=args.lr,
    decay=args.decay,
    ls=args.ls)
