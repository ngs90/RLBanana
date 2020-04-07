import numpy as np

# SUM Tree functions

class Node:
    # Based on: https://adventuresinmachinelearning.com/sumtree-introduction-python/
    def __init__(self, left, right, is_leaf: bool=False, idx=None, insertion_time=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx
        self.insertion_time = insertion_time
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self
        
        #self.value = sum(n.value for n in (left, right) if n is not None)
        
    @classmethod
    def create_leaf(cls, value, idx, insertion_time):
        leaf = cls(None, None, is_leaf=True, idx=idx, insertion_time=insertion_time)
        leaf.value = value
        return leaf 
    
    
def create_tree(inp: list, insertion_times: list):
    nodes = [Node.create_leaf(v,i,t) for i, (v, t) in enumerate(zip(inp, insertion_times))]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]
    return nodes[0], leaf_nodes


def retrieve(value: float, node: Node):
    if node.is_leaf:
        return node
    
    if node.left.value >= value:
        return retrieve(value, node.left)
    else:
        return retrieve(value - node.left.value, node.right)
    
def update(node: Node, new_value: float, new_insertion_time=None):
    change = new_value - node.value
    
    node.value = new_value
    
    if new_insertion_time:
        node.insertion_time = new_insertion_time
    
    propagate_changes(change, node.parent)
    
def propagate_changes(change: float, node: Node):
    node.value = node.value + change
    
    if node.parent is not None:
        propagate_changes(change, node.parent)


if __name__ == '__main__': 
           
    # #INPUT = [1,2,3,4,6,15,4,4]
    INPUT = [1,4,2,3]
    insertion_times = [0,1,2,3]
    root_node, leaf_nodes = create_tree(INPUT, insertion_times)
    def demonstrate_sampling(root_node: Node):
        tree_total = root_node.value
        iterations = 100000
        selected_vals = []
        selected_idxs = []
        selected_insertion_times = []
        for i in range(iterations):
            rand_val = np.random.uniform(0, tree_total)
            leaf_node = retrieve(rand_val, root_node)
            selected_val = leaf_node.value
            selected_idx = leaf_node.idx
            selected_insertion_time = leaf_node.insertion_time
            selected_vals.append(selected_val)
            selected_idxs.append(selected_idx)
            selected_insertion_times.append(selected_insertion_time)

        return selected_vals, selected_idxs, selected_insertion_times
    selected_vals, selected_idxs, selected_insertion_times = demonstrate_sampling(root_node)
    #print(selected_vals[:100])
    # the below print statement should output ~4
    print(f"Should be ~4: {sum([1 for x in selected_vals if x == 4]) / sum([1 for y in selected_vals if y == 1])}")
    # the below print statement should output ~1.5
    print(f"Should be ~1.5: {sum([1 for x in selected_idxs if x == 3]) / sum([1 for y in selected_idxs if y == 2])}")
    # the below print statement should output ~2
    print(f"Should be ~2: {np.mean([t for x, t in zip(selected_idxs, selected_insertion_times) if x == 2])}")
