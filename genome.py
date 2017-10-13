from random import choice,random,randint
from activations import ActivationFunctions
from copy import deepcopy

act = ActivationFunctions()

class Genome:
    def __init__(self,inp_n,out_n):
        self.node = {}
        self.connection = []
        self.inp_n = inp_n
        self.out_n = out_n
        for i in range(inp_n):
            self.add_node(i)
        for o in range(out_n):
            self.add_node(inp_n+o,activation='sigmoid')
        for i in range(inp_n):
            for o in range(out_n):
                self.add_connection(i,inp_n+o,random()*2-1.,True)
        self.build()
    def add_node(self,n,activation=None):
        #activation = 'sigmoid'
        self.node.update({n:[act.set(activation),set([])]})
    def add_connection(self,inp_node,out_node,weight,enabled):
        for i,con in enumerate(self.connection):
            if con[0] == inp_node and con[1] == out_node:
                self.connection[i][2] = weight
                return
        self.connection.append([inp_node,out_node,weight,enabled])
        self.node[out_node][1].add(inp_node)
    def build(self):
        s = set([])
        s_o = set(i+self.inp_n for i in range(self.out_n))
        connection_ = []
        for i in range(self.inp_n):
            s.add(i)
        count = 1
        while count != 0: #len(s) != len(self.node):
            count = 0
            for con in self.connection:
                if con in connection_:
                    continue
                if self.node[con[0]][1].issubset(s):
                    connection_.append(con)
                    count += 1
            for j,(k,v) in enumerate(self.node.items()):
                if v[1].issubset(s):
                    s.add(k)
        self.connection = deepcopy(connection_)
    def forward(self,x):
        node = {k:0. for k,v in self.node.items()}
        for i in range(self.inp_n):
            node[i] = x[i]
        for con in self.connection:
            node[con[1]] += node[con[0]] * con[2]
        node = [v for k,v in node.items()]
        return node[self.inp_n:self.inp_n+self.out_n]
    def mutate_add_node(self,n):
        #print self.node
        #print self.connection
        c = randint(0,len(self.connection)-1)
        con = self.connection[c]
        self.add_node(n,set([con[0]]))
        self.add_connection(con[0],n,random()*2-1.,True)
        self.add_connection(n,con[1],random()*2-1.,True)
        self.node[con[1]][1].remove(con[0])
        self.node[con[1]][1].add(n)
        del self.connection[c]
    def mutate_delete_node(self):
        available_nodes = [k for k in self.node.keys() if self.inp_n+self.out_n <= k]
        if len(available_nodes) == 0:
            return
        del_key = choice(available_nodes)
        deletes = []
        for i,c in enumerate(self.connection):
            if del_key == c[0] or del_key == c[1]:
                if not self.check_output(c[0],c[1]):
                    return
                deletes.append(i)
        for i in reversed(deletes):
            self.node[self.connection[i][1]][1].remove(self.connection[i][0])
            del self.connection[i]
        del self.node[del_key]
    def mutate_add_connection(self):
        inp_n = 0
        out_n = 0
        while self.check_cycle(self.node.keys()[inp_n],self.node.keys()[out_n]):
            inp_n = self.inp_n
            while self.inp_n <= inp_n and inp_n < self.inp_n+self.out_n:
                inp_n = randint(0,len(self.node)-1)
            out_n = randint(self.inp_n,len(self.node)-1)
        self.add_connection(self.node.keys()[inp_n],self.node.keys()[out_n],random()*2-1.,True)
    def mutate_delete_connection(self):
        if self.connection:
            key = randint(0,len(self.connection)-1)
            for o in range(self.out_n):
                if self.connection[key][1] == self.inp_n+self.out_n-1+o and len(self.node[self.inp_n+self.out_n-1+o][1]) == 1:
                    return
            if not self.check_output(self.connection[key][0],self.connection[key][1]):
                return
            self.node[self.connection[key][1]][1].remove(self.connection[key][0])
            del self.connection[key]
    def check_cycle(self,inp_n,out_n):
        if inp_n == out_n:
            return True
        visited = {out_n}
        while True:
            num_added = 0
            for con in self.connection:
                if con[0] in visited and con[1] not in visited:
                    if con[1] == inp_n:
                        return True
                    visited.add(con[1])
                    num_added += 1
            if num_added == 0:
                return False
    def check_output(self,inp_n,out_n):
        visited = [i for i in range(inp_n)]
        for c in self.connection:
            if c[0] == inp_n and c[1] == out_n:
                continue
            if c[0] in visited:
                visited.append(c[1])
        for n in range(self.out_n):
            if self.inp_n + self.out_n - 1 + n not in visited:
                return False
        return True
