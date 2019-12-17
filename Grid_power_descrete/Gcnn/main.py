import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from models.graphcnn import GraphCNN

criterion = nn.CrossEntropyLoss()
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

class Policy():

    def __init__(self):
        self.model = GraphCNN(5, 2, 7, 64, 2, 0.5, False, 'sum', 'sum', device).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)


    def train(self, state_graphs,nstate_graph,epoch=5):
        self.model.train()
        total_iters = 10
        pbar = tqdm(range(total_iters), unit='batch')
        loss_accum = 0
        for pos in pbar:
            selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
            use_state_graph = [state_graphs[idx] for idx in selected_idx]
            use_nstate_graph = [nstate_graphs[idx] for idx in selected_idx]
            output = self.model(use_state_graph)
            noutput= self.model(use_nstate_graph)
            #compute loss
            loss = criterion(output, noutput)
            #backprop
            if optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            loss = loss.detach().cpu().numpy()
            loss_accum += loss
            #report
            pbar.set_description('epoch: %d' % (epoch))

        average_loss = loss_accum/total_iters
        print("loss training: %f" % (average_loss))

        return average_loss

    def pass_data_iteratively(self, graphs, minibatch_size = 64):
        self.model.eval()
        output = []
        idx = np.arange(len(graphs))
        for i in range(0, len(graphs), minibatch_size):
            sampled_idx = idx[i:i+minibatch_size]
            if len(sampled_idx) == 0:
                continue
            output.append(self.model([graphs[j] for j in sampled_idx]).detach())
        return torch.cat(output, 0)

    def test(self,test_graphs, epoch):
        self.scheduler.step()
        self.model.eval()
        output = self.pass_data_iteratively(test_graphs)
        pred = output.max(1, keepdim=True)[1]
        return pred

agent=Policy()
test_data=np.random.random((1,7))
t_data=torch.from_numpy(test_data).float().to(device)
print(test_data)
agent.test(t_data,1)