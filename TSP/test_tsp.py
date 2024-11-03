import os

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from TSP.Actor import Model
from TSP.create_tsp_instance import reward1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rollout(model, dataset, n_nodes):
    model.eval()
    tour = []

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, _ = model(bat, n_nodes, True)
            tour.append(cost.cpu().numpy().tolist())
            cost = reward1(bat.x, cost.detach(), n_nodes)
        return cost.cpu()

    total_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return total_cost, tour


def evaluate(valid_loader, n_node):
    folder = "trained"
    actor = Model(2, 128, 1, 16, conv_layers=4).to(device)
    filepath = os.path.join(folder, "%s" % n_node)

    if os.path.exists(filepath):
        path1 = os.path.join(filepath, "actor.pt")
        actor.load_state_dict(torch.load(path1, device, weights_only=True))
    # -------------------------------------------------------------Greedy
    cost, tour = rollout(actor, valid_loader, n_node)
    cost = cost.mean()
    print("Problem: TSP" "%s" % n_node, "/ Average distance:", cost.item())
    return cost
    # -------------------------------------------------------------


def test(n_node):
    datas = []
    if n_node == 20 or n_node == 50 or n_node == 100:
        node_ = np.loadtxt(
            "./test_data/tsp{}_test_data.csv".format(n_node),
            dtype=np.float32,
            delimiter=",",
        )
        batch_size = 128

    else:
        print("Please enter 20, 50 or 100")
        return
    node_ = node_.reshape(-1, n_node, 2)

    # Calculate the distance matrix
    def c_dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5

    # edges = torch.zeros(n_nodes,n_nodes)

    data_size = node_.shape[0]
    edges = np.zeros((data_size, n_node, n_node, 1))
    for k, data in enumerate(node_):
        for i, (x1, y1) in enumerate(data):
            for j, (x2, y2) in enumerate(data):
                d = c_dist((x1, y1), (x2, y2))
                edges[k][i][j][0] = d
    edges_ = edges.reshape(data_size, -1, 1)

    edges_index = []
    for i in range(n_node):
        for j in range(n_node):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)
    for i in range(data_size):
        # node,edge = create_instance(n_nodes)
        data = Data(
            x=torch.from_numpy(node_[i]).float(),
            edge_index=edges_index,
            edge_attr=torch.from_numpy(edges_[i]).float(),
        )
        datas.append(data)
    # print(datas)
    print("Data created")
    dl = DataLoader(datas, batch_size=batch_size)
    evaluate(dl, n_node)


"please enter 20, 50 or 100"
test(20)
