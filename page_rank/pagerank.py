# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:23:25 2018

@author: 杨扬
"""

import numpy as np
import time

#WikiData数据路径
src = './WikiData.txt'
#分块数据存储及读取基础路径
outBase = './output/output'
#分块儿的大小，当设置为最大值103689时即不分块
block_size = 2000
#block_size = 103689

#导入数据
def load_data(path):
    f = open(path, 'r')
    data = []
    for line in f:
        x, y = line.split()
        data.append([int(x), int(y)])
    data = np.array(data)
    f.close()
    return data

#生成各节点ID列表及最大节点ID
def get_info(path, num_blocks):
    index = []
    max_node = 0
    for i in range(num_blocks):
        block = path + str(i) + '.txt'
        f = open(block, 'r')
        for line in f:
            x, y = line.split()
            if max_node < int(x):
                max_node = int(x)
            if max_node < int(y):
                max_node = int(y)
            index.append(int(x))
            index.append(int(y))
        f.close()
    index = list(np.unique(index))
    return index, max_node

#获取各节点出度
def get_out_degree(path, num_blocks, index, sign):
    out = np.zeros(len(index))
    for i in range(num_blocks):
        block = path + str(i) + '.txt'
        data = np.loadtxt(block)
        for j in range(len(data)):
            out[int(sign[int(data[j][0])])] += 1
    return out

#获取节点的映射关系及r的初始值
def preprocess(index, max_node):
    #对节点ID进行编码
    sign = np.zeros(max_node + 1)
    sign = list(sign)
    for i in range(len(index)):
        sign[index[i]] = i
    #初始化page_rank值
    r = np.ones(len(index))/len(index)
    return sign, r

#根据数据读取节点的链接关系，以稀疏形式保存，只存储非零的链接
def get_edges(data, sign):
    edges = np.zeros(data.shape)
    for i in range(len(data)):
        edges[i]=[sign[data[i][0]], sign[data[i][1]]]
    return edges

#对数据进行分块，保存至指定的路径
def block_data(path, splitLen):
    splitLen = 2000
	#保存分块数据的基路径
    outputBase = './output/output'
	#输入数据的路径
    inputs = open('./WikiData.txt', 'r')
    count = 0
    at = 0
    dest = None
    for line in inputs:
        if count % splitLen == 0:
            if dest: 
                dest.close()
            dest = open(outputBase + str(at) + '.txt', 'w')
            at += 1
        dest.write(line)
        count += 1
    return at

#pagerank
def block_stripe_page_rank(path, num_blocks, index, sign, r, out, beta):
    r_old = r
    while True:
        r_new = np.zeros(len(index))
        #读取数据执行pagerank算法，每次读取一个分块的数据，以达到节省内存的效果
        for i in range(num_blocks):
            block = path + str(i) + '.txt'
            data = load_data(block)
            edges = get_edges(data, sign)
            for edge in edges:
                r_new[int(edge[1])] += r_old[int(edge[0])] * beta/out[int(edge[0])]
        r_sum=sum(r_new)
        r_sub=np.ones(len(index))*(1-r_sum)/len(index)
        r_cur = r_new + r_sub
        s = np.sqrt(sum((r_cur - r_old)**2))
        #当改变值小于1e-8时，判断pagerank收敛，跳出循环
        if s <= 1e-8:
            r_old=r_cur
            break
        else:
            r_old=r_cur
    return r_old

#获取top100的节点及其pagerank值
def get_top(num, r):
    r_index = r.argsort()[::-1][:100]
    r.sort()
    top_r = r[::-1][:100]
    top_index = np.zeros(100)
    for i in range(100):
        top_index[i] = sign.index(r_index[i])
    top_index = [int(i) for i in top_index]
    for i in range(100):
        print (top_index[i], top_r[i])
    return top_index, top_r

def write_out(top_index, top_r):
    f = open('./result.txt', 'w')
    for i in range(len(top_index)):
        f.write(str(top_index[i]) + '  ' + str(top_r[i]) + '\n')
    f.close()
    return

if __name__ == '__main__':
    start = time.clock()
    num_blocks = block_data(src, block_size)
    index, max_node = get_info(outBase, num_blocks)
    sign, r = preprocess(index, max_node)
    out = get_out_degree(outBase, num_blocks, index, sign)
    r_final = block_stripe_page_rank(outBase, num_blocks, index, sign, r, out, 0.85)
    print('-----------------------')
    top_index, top_r = get_top(100, r_final)
    write_out(top_index, top_r)
    end = time.clock()
    print('time cost: ', str(end - start), 's')
