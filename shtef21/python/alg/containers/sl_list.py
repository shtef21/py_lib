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


    def reverse(self):
        if self.head is None:
            return
        
        prev = None
        curr = self.head
        next = None

        while curr != None:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        self.head = prev


    def add(self, val):
        node = self.Node(val, None)
        
        if self.head is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node


    def delete_at(self, index):
        if self.head is not None:
            if index == 0:
                self.head = self.head.next
                return
            idx_next = 1
            curr = self.head.next
            while curr is not None:
                if idx_next == index:
                    curr.next = curr.next.next if curr.next.next is not None else None
                    return
                curr = curr.next
                idx_next += 1

    
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

        while curr is not None:
            if curr_idx == idx:
                return curr.val
            curr = curr.next
            curr_idx += 1

        # TODO: raise index error

    
    def __setitem__(self, idx, new_val):
        curr_idx = 0
        curr = self.head

        while curr is not None:
            if curr_idx == idx:
                curr.val = new_val
                return
            curr = curr.next
            curr_idx += 1

        # TODO: raise index error

    
    def __contains__(self, item):
        curr = self.head

        while curr is not None:
            if curr.val == item:
                return True
            curr = curr.next

        return False
    
    
    def __iter__(self):
        self.iter_curr = self.head
        return self


    def __next__(self):
        if self.iter_curr is not None:
            result = self.iter_curr.val
            self.iter_curr = self.iter_curr.next
            return result
        raise StopIteration


list_obj = sl_list([1, 2, 3])

print(f'''
    len(list) = { len(list_obj) }
    list[2] = { list_obj[2] }
    list[100] = { list_obj[100] }
    3 in list = { 3 in list_obj }
    33 in list = { 33 in list_obj }
''')
      
list_obj[2] = 44
      
for node in list_obj:
    print('iter:', node)
      
print('new list...')
list_obj = sl_list([el for el in range(10)])
      
for node in list_obj:
    print('iter:', node)
