import numpy as np
import json
import torch
from torch_geometric.data import Data
from scitsr.eval import json2Relations, eval_relations
from scitsr.relation import Relation

class Graph_to_Adjacency_relationship(object):
    def __init__(self, data):
        self.data = data

        # 获取判定为同行/同列的边
        self.same_row = self.delete_0_edge(np.array(self.data.y_row_pre), np.array(self.data.new_edge_index))
        self.same_col = self.delete_0_edge(np.array(self.data.y_col_pre), np.array(self.data.new_edge_index))

        # 规则修正入度大于1的同行边：一个节点的入度大于1时，保留距离最近的节点
        self.row_indegree = {}
        for i in range(len(self.same_row)):
            if self.same_row[i, 1] in self.row_indegree:
                self.row_indegree[self.same_row[i, 1]].append(self.same_row[i, 0])
            else:
                self.row_indegree[self.same_row[i, 1]] = [self.same_row[i, 0]]
        rectify_row_indegree = self.fliter_indegree_wrong(self.row_indegree, self.same_row)
        self.rectify_same_row = self.dict_to_edge(rectify_row_indegree)
        print(self.rectify_same_row)

        # 规则修正入度大于1的同列边：一个节点的入度大于1时，保留距离最近的节点
        self.col_indegree = {}
        for i in range(len(self.same_col)):
            if self.same_col[i, 1] in self.col_indegree:
                self.col_indegree[self.same_col[i, 1]].append(self.same_col[i, 0])
            else:
                self.col_indegree[self.same_col[i, 1]] = [self.same_col[i, 0]]
        rectify_col_indegree = self.fliter_indegree_wrong(self.col_indegree, self.same_col)
        self.rectify_same_col = self.dict_to_edge(rectify_col_indegree)
        print(self.rectify_same_col)

        self.ret = self.edge_to_relation(self.rectify_same_row, self.rectify_same_col)

    def delete_0_edge(self, preds, edge_index):
        edges_new = []
        for i in range(len(preds)):
            if preds[i] == 1:
                edges_new.append([edge_index[0, i], edge_index[1, i]])

        index = np.lexsort([np.array(edges_new)[:, 1], np.array(edges_new)[:, 0]])
        return np.array(edges_new)[index, :]

    def fliter_indegree_wrong(self, indegree_dict, edge):
        for i in range(len(edge)):
            if len(indegree_dict[edge[i, 1]]) > 1:
                MIN_DISTANCE = float("inf")
                MIN_DISTANCE_POINT = []
                for j in range(len(indegree_dict[edge[i, 1]])):
                    tmp_distance = self.cal_distance(self.data.pos[edge[i, 1]], self.data.pos[indegree_dict[edge[i, 1]][j]])
                    if tmp_distance < MIN_DISTANCE:
                        MIN_DISTANCE = tmp_distance
                        MIN_DISTANCE_POINT = [indegree_dict[edge[i, 1]][j]]
                indegree_dict[edge[i, 1]] = MIN_DISTANCE_POINT
        return indegree_dict

    def cal_distance(self, point1, point2):
        return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

    def dict_to_edge(self, indegree_dict):
        edge = []
        for k, v in indegree_dict.items():
            edge.append([v[0], k])
        return edge

    def edge_to_relation(self, same_row_edge, same_col_edge):
        ret = []
        DIR_HORIZ = 1
        DIR_VERT = 2

        for i in range(len(same_row_edge)):
            from_id, to_id = same_row_edge[i][0], same_row_edge[i][1]
            ret.append(Relation(from_text=np.array(self.data.text)[from_id],
                                to_text=np.array(self.data.text)[to_id],
                                direction=DIR_HORIZ,
                                from_id=from_id,
                                to_id=to_id,
                                no_blanks=0))

        for i in range(len(same_col_edge)):
            from_id, to_id = same_col_edge[i][0], same_col_edge[i][1]
            ret.append(Relation(from_text=np.array(self.data.text)[from_id],
                                to_text=np.array(self.data.text)[to_id],
                                direction=DIR_VERT,
                                from_id=from_id,
                                to_id=to_id,
                                no_blanks=0))

        return ret

    # def find_row(self, pos):
    #     cell_num = len(pos)
    #     id = [[i] for i in range(cell_num)]
    #     pos = np.concatenate((pos, id), axis=1)
    #
    #     index = np.lexsort([pos[:, 1], pos[:, 0]])
    #     pos = pos[index, :]
    #
    #     for i in range(cell_num):
    #         tmp_path = []
    #         self.dfs(pos[i][2], tmp_path)
    #
    #     # print(same_row_index)
    #     # print(edge)
    #
    # def dfs(self, node, tmp_path):
    #     tmp_path.append(node)
    #     if node not in self.row_edge or len(self.row_edge[node]) == 0:
    #         if len(tmp_path) != 1:
    #             self.row.append(tmp_path[:])
    #             tmp_path.pop()
    #         return
    #
    #     for _ in range(len(self.row_edge[node])):
    #         next_node = self.row_edge[node].pop()
    #         self.dfs(next_node, tmp_path)
    #
    #     tmp_path.pop()

if __name__ == '__main__':
    x = torch.FloatTensor(list(range(12)))
    pos = [[2, 2], [4, 2], [6, 2], [8, 2],
           [2, 5], [4, 5], [6, 5], [8, 5],
           [2, 8], [4, 8], [6, 8], [8, 8]]
    text = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    pos = torch.FloatTensor(pos)
    data = Data(x=x, pos=pos)
    # new_edge_index = [[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,  7,  8, 9,  10],
    #                   [1, 4, 2, 5, 6, 3, 6, 7, 6, 7, 5, 8, 6, 9, 7, 10, 11, 9, 10, 11]]
    new_edge_index = [[0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6,  7,  8, 9,  10],
                      [1, 4, 2, 5, 3, 6, 7, 5, 8, 6, 9, 7, 10, 11, 9, 10, 11]]
    y_row_pre =       [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,  0,  1, 1,  1]
    y_col_pre =       [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1,  1,  0, 0,  0]

    data.new_edge_index = torch.LongTensor(new_edge_index)
    data.y_row_pre = torch.LongTensor(y_row_pre)
    data.y_col_pre = torch.LongTensor(y_col_pre)
    data.text = text
    # print(data)

    x1 = torch.FloatTensor([4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11])

    pos1 = [[2, 5], [4, 5], [6, 5], [8, 5],
           [2, 2], [4, 2], [6, 2], [8, 2],
           [2, 8], [4, 8], [6, 8], [8, 8]]
    text1 = ["4", "5", "6", "7", "0", "1", "2", "3", "8", "9", "10", "11"]
    pos1 = torch.FloatTensor(pos1)
    data1 = Data(x=x1, pos=pos1)
    new_edge_index1 = [[0, 0, 1, 1, 2, 2,  3,  4, 4, 5, 5, 6, 6, 7, 8, 9,  10],
                       [1, 8, 2, 9, 3, 10, 11, 0, 5, 1, 6, 2, 7, 3, 9, 10, 11]]
    y_row_pre1 =       [1, 0, 1, 0, 1, 0,  0,  0, 1, 0, 1, 0, 1, 0, 1, 1,  1]
    y_col_pre1 =       [0, 1, 0, 1, 0, 1,  1,  1, 0, 1, 0, 1, 0, 1, 0, 0,  0]

    # x1 = torch.FloatTensor([4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11])
    # pos1 = [[2, 5], [4, 5], [6, 5], [8, 5],
    #        [2, 2], [4, 2], [6, 2], [8, 2],
    #        [2, 8], [4, 8], [6, 8], [8, 8]]
    # text1 = ["4", "5", "6", "7", "0", "1", "2", "3", "8", "9", "10", "11"]
    # pos1 = torch.FloatTensor(pos1)
    # data1 = Data(x=x1, pos=pos1)
    # new_edge_index1 = [[0, 0, 0, 1, 1, 1, 2, 2, 2,  3, 3,  4, 5, 6, 8, 9,  10],
    #                    [1, 8, 4, 2, 5, 9, 3, 6, 10, 7, 11, 5, 6, 7, 9, 10, 11]]
    # y_row_pre1 =       [1, 0, 0, 1, 0, 0, 1, 0, 0,  0, 0,  1, 1, 1, 1, 1,  1]
    # y_col_pre1 =       [0, 1, 1, 0, 1, 1, 0, 1, 1,  1, 1,  0, 0, 0, 0, 0,  0]

    data1.new_edge_index = torch.LongTensor(new_edge_index1)
    data1.y_row_pre = torch.LongTensor(y_row_pre1)
    data1.y_col_pre = torch.LongTensor(y_col_pre1)
    data1.text = text1

    gt_relations = Graph_to_Adjacency_relationship(data).ret
    res_relations = Graph_to_Adjacency_relationship(data1).ret
    # print(res_relations)

    precision, recall = eval_relations(gt=[gt_relations], res=[res_relations], cmp_blank=True)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("P: %.4f, R: %.4f, F1: %.4f" % (precision, recall, f1))
