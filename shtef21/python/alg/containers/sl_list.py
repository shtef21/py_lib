# Singly-linked list

class sl_list():


    class Node():
        def __init__(self, val, next):
            self.val = val
            self.next = next


    def __init__(self, init_arr = None):
        self.head = None
        self.tail = None

        if type(init_arr) is list:
            for el in init_arr:
                self.add(el)


    def add(self, val):
        node = self.Node(val, None)
        
        if self.head is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    
    def __len__(self):
        len = 0
        curr = self.head

        while curr is not None:
            len += 1
            curr = curr.next
        return len


    def __getitem__(self, idx):
        curr_idx = 0
        curr = self.head
        val = None

        while curr is not None and curr_idx != idx:
            curr = curr.next
            val = curr.val
            curr_idx += 1

        return val


list = sl_list([1, 2, 3])

print(f'''
    len(list) = { len(list) }
''')
