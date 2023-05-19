
class node:

    def __init__(self, value):
        self.value = value
        self.left = None
        self.left_height = 0
        self.right = None
        self.right_height = 0


class avl_tree:

    def __init__(self):
        self.head = node(3)
        self.height = 0
        

def print_tree(node):
    if node is None:
        return
    print_tree(node.left)
    print(node.value, end=' ')
    print_tree(node.right)


tree = avl_tree()
print_tree(tree.head)
print()
