from typing import List, Optional


class TreeNode:
    def __init__(self):
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
        self.index: int = -1

    def partition(self):
        if self.left is None or self.right is None:
            return [[self.index]]

        left_nodes = self.left.partition()
        right_nodes = self.right.partition()

        all_partitions = []
        for l_node in left_nodes:
            for r_node in right_nodes:
                h = l_node.copy()
                h.extend(r_node)
                all_partitions.append(h)

        all_partitions.append([self.index])
        return all_partitions


def create_tree(depth: int) -> TreeNode:
    node = TreeNode()
    if depth == 0:
        return node
    node.left = create_tree(depth-1)
    node.right = create_tree(depth-1)

    return node


def add_indices(node: TreeNode) -> TreeNode:
    nodes: List[TreeNode] = [node]
    i = 0
    while nodes:
        n = nodes.pop(0)
        n.index = i

        if n.left != None:
            nodes.append(n.left)
        if n.right != None:
            nodes.append(n.right)
        i += 1

    return node


if __name__ == "__main__":
    depth = 2
    root = create_tree(depth)
    root = add_indices(root)

    print(root.partition())