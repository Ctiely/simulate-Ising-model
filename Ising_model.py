#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: clytie
@school: Renmin University of China
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class Ising_model(object):
    """
    simulate Ising model
    """
    def __init__(self, NODE_NUM):
        x = np.random.binomial(1, 0.5, size=NODE_NUM)
        x[x == 0] = -1
        self.x = x
        self.node_num = NODE_NUM
        self.theta_edge = None
        self.theta_node = None
        
    def get_random(self, p):
        randnum = np.random.binomial(1, p)
        if randnum:
            return(1)
        else:
            return(-1)
            
    @property
    def set_diag_zero(self):
        node_num = self.node_num
        for i in np.arange(node_num):
            self.theta_edge[i][i] = 0
    
    @property
    def set_symmetry(self):
        MM = self.theta_edge
        M = np.triu(MM)
        M += M.T - np.diag(M.diagonal())
        self.theta_edge = M
        
    def get_param_edge(self, param_dict=None):
        NODE_NUM = self.node_num
        theta_edge = np.zeros((NODE_NUM, NODE_NUM))
        if param_dict is None:
            theta_edge = np.random.normal(size=(NODE_NUM, NODE_NUM))
            self.theta_edge = theta_edge
            self.set_diag_zero
            self.set_symmetry
            return(self.theta_edge)
        for index in param_dict:
            theta_edge[index] = param_dict[index]
        self.theta_edge = theta_edge
        self.set_diag_zero
        self.set_symmetry
        return(self.theta_edge)
    
    def get_param_node(self, param_dict=None):
        NODE_NUM = self.node_num
        theta_node = np.zeros(NODE_NUM)
        if param_dict is None:
            theta_node = np.random.normal(size=NODE_NUM)
            self.theta_node = theta_node
            return(theta_node)
        for index in param_dict:
            theta_node[index] = param_dict[index]
        self.theta_node = theta_node
        return(theta_node)
    
    def gibbs_i(self, i):
        x = self.x.copy()
        tmp_x = x.copy()
        tmp_x[i] = 1
        tmp_x_2 = np.dot(tmp_x.reshape(-1, 1), tmp_x.reshape(1, -1))
        p_1 = np.sum(tmp_x * self.theta_node) + np.sum(tmp_x_2 * self.theta_edge)
        tmp_x[i] = -1
        tmp_x_2 = np.dot(tmp_x.reshape(-1, 1), tmp_x.reshape(1, -1))
        p_0 = np.sum(tmp_x * self.theta_node) + np.sum(tmp_x_2 * self.theta_edge)
        p = 1 / (1 + np.exp(p_0 - p_1))
        #print(p)
        x[i] = self.get_random(p)
        self.x = x
        #return(x)
    
    def gibbs(self, N):
        if self.theta_edge is None:
            self.get_param_edge()
        if self.theta_node is None:
            self.get_param_node()
        n = self.node_num
        for j in np.arange(N):
            if (j + 1) % 10 == 0:
                print("epoch{}: start".format(j + 1))
            for i in np.arange(n):
                self.gibbs_i(i)
            if (j + 1) % 10 == 0:
                print("finished.")
            
    @property
    def check_square(self):
        node_num = self.node_num
        t = int(np.sqrt(node_num) + 1e-4)
        return(t ** 2 == node_num)
    
    def matshow(self):
        """
        1: white
        -1: black
        只有一种数字就选择默认颜色
        """
        x = self.x
        if not self.check_square:
            raise Exception("node number {} isn't square number.".format(self.node_num))
        m = int(np.sqrt(self.node_num) + 1e-4)
        if len(np.unique(x)) == 1:
            plt.matshow(x.reshape((m, m)))
            plt.show()
        else:
            
            cmap = colors.ListedColormap(["black", "white"])
            plt.matshow(x.reshape((m, m)), cmap=cmap)
            plt.show()
        
if __name__ == "__main__":    
    NODE_NUM = 625
    #625个变量有两两之间具负相关关系
    model = Ising_model(NODE_NUM)
    param_dict = {}
    for i in np.arange(NODE_NUM):
        for j in np.arange(NODE_NUM):
            if i >= j:
                continue
            param_dict[(i, j)] = -np.random.rand() * 100
    model.get_param_edge(param_dict)
    plt.matshow(model.theta_edge)
    model.gibbs(50)
    model.matshow()    
    #随机变量完全独立
    model = Ising_model(NODE_NUM)
    model.get_param_edge({})
    model.gibbs(50)
    model.matshow()
    #625个变量具有正相关关系
    model = Ising_model(NODE_NUM)
    param_dict = {}
    for i in np.arange(NODE_NUM):
        for j in np.arange(NODE_NUM):
            if i >= j:
                continue
            param_dict[(i, j)] = np.random.rand() / 300
    model.get_param_edge(param_dict)
    plt.matshow(model.theta_edge)
    model.gibbs(50)
    model.matshow()    
    
            

        
        
        
        
        
        