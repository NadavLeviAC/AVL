# username - nadavlevi
# id1      - 314831017
# name1    - nadav levi
# id2      - 313354508
# name2    - or levy

"""A class represnting a node in an AVL tree"""
import random


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type value: str
    @param value: data of your node
    """

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1  # Balance factor
        self.size = 0



    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """

    def getLeft(self):
        return self.left

    """returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        return self.right

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value

    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        return self.value

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        return self.height


    """returns the size

    @rtype:int
     @return the size of self, o if node is virtual
    """


    def getSize(self):
        return self.size

    """returns the balance factor of the node
    
    @rtype:int
    returns: balance factor"""
    def getBalanceFactor(self):
        return self.left.getHeight() - self.right.getHeight()

    """sets left child
    
      @type node: AVLNode
      @param node: a node
      """


    def setLeft(self, node):
        self.left = node


    """sets right child
    
    @type node: AVLNode
    @param node: a node
    """


    def setRight(self, node):
        self.right = node


    """sets parent
    
    @type node: AVLNode
    @param node: a node
    """


    def setParent(self, node):
        self.parent = node


    """sets value
    
    @type value: str
    @param value: data
    """


    def setValue(self, value):
        self.value = value


    """sets the balance factor of the node
    
    @type h: int
    @param h: the height
    """


    def setHeight(self, h):
        self.height = h;


    """sets the size of the node
    
    @type s: int
    @param s: the size
    """


    def setSize(self, s):
        self.size = s;


    """returns whether self is not a virtual node 
    
    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """


    def isRealNode(self):
        return self.size > 0

    """updates the size and height of the node
    
    @rtype = None
    returns nothing"""
    def updateSizeHeight(self):
        self.size = 1 + self.right.size + self.left.size
        self.height = 1 + max(self.right.getHeight(),self.left.getHeight())

    """
    A class implementing the ADT list, using an AVL tree.
    """
    def deleteNode(self):
        parent = self.getParent()
        if (parent == None):
            x = 0
        elif (parent.getRight() == self):
            parent.setRight(AVLNode(None))
        elif(parent.getLeft() == self):
            parent.setLeft(AVLNode(None))
        self.setParent(None)

    def buildLeaf(self):
        self.size = 1
        self.height=0
        self.right = AVLNode(None)
        self.left = AVLNode(None)

class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self):
        self.size = 0
        self.root = None
        self.firstItem = None
        self.lastItem = None

    # add your fields here

    """returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        return self.root == None

    """retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    @Complexity: O(log(n)) - Select cost is log(n)
    """

    def retrieve(self, i):
        return self.Select(self.root,i+1).getValue()

    """inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    @complexity: O(log(n)) - cost of selecting next node, finding predeseccor,fixing height,size,BF is all log(n)
    """

    def insert(self, i, val):
        newNode = AVLNode(val)
        newNode.height = 0
        newNode.size = 1
        newNode.setLeft(AVLNode(None))
        newNode.setRight(AVLNode(None))
        if self.size == 0:
            self.root = newNode
        elif i == self.size:
            maxi = self.maximum(self.root)
            maxi.setRight(newNode)
            newNode.setParent(maxi)
            self.lastItem = newNode

        elif i < self.size:
            next_node =self.Select(self.root,i+1)
            if not next_node.getLeft().isRealNode():
                next_node.setLeft(newNode)
                newNode.setParent(next_node)
            else:
                prede = self.predecessor(next_node)
                prede.setRight(newNode)
                newNode.setParent(prede)

        self.size += 1
        self.fixHeightSizeInsert(newNode)
        self.fixBalanceFactorInsert(newNode)
        if(i == 0):
            self.firstItem = newNode



    """deletes the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    @Complexity: cost of selecting the Node, Find successor, and fixing the tree is all log(n)
    """

    def delete(self, i):
        deletedNode = self.Select(self.root,i+1)
        if(i==0): #updates self.firstItem
            if (self.size > 1):
                self.firstItem = self.Select(self.root,i+2)
            else:
                self.firstItem = None
                self.root = None
                self.lastItem = None
        if(i+1==self.size): #updates self.max
            if self.size >1 : self.lastItem = self.predecessor(deletedNode)
            else: self.lastItem = None
        if(deletedNode.getLeft().isRealNode() and deletedNode.getRight().isRealNode()):
            successor = self.Select(self.root,i+2)
            deletedNode.setValue(successor.getValue())
            parent = successor.getParent()
            deletedNode = successor
            #successor.deleteNode()
            #self.delete(i+1)
            #return

        if(deletedNode.getLeft().isRealNode() or deletedNode.getRight().isRealNode()):
            parent = deletedNode.getParent()
            if parent == None:
                self.root = deletedNode
                deletedNode.setParent(None)
            elif(parent.getRight() == deletedNode):
                if(deletedNode.getLeft().isRealNode()):
                    parent.setRight(deletedNode.getLeft())
                else:
                    parent.setRight(deletedNode.getRight())
                parent.getRight().setParent(parent)
            else:
                if (deletedNode.getLeft().isRealNode()):
                    parent.setLeft(deletedNode.getLeft())
                else:
                    parent.setLeft(deletedNode.getRight())
                parent.getLeft().setParent(parent)

        else:
            parent = deletedNode.getParent()
            deletedNode.deleteNode()
        self.size -= 1
        self.fixHeightSizeDelete(parent)
        self.fixBalanceFactorDelete(parent)





    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    @complexity: O(1) - returning a variable is O(1)
    """

    def first(self):
        return self.firstItem

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    @complexity: O(1) - returning a variable is O(1)
    """

    def last(self):
        return self.lastItem

    """returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    @Complexity: same as in Order scan O(n)
    """

    def listToArray(self):
        if (self.root == None):
            return []
        def listToArrayRec(node):
            if not node.isRealNode():
                return []
            return listToArrayRec(node.getLeft()) + [str(node.getValue())] + listToArrayRec(node.getRight())
        return listToArrayRec(self.root)

    """returns the size of the list 

    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        return self.size

    """sort the info values of the list

    @rtype: list
    @returns: an AVLTreeList where the values are sorted by the info of the original list.
    @complexity: mergesort cost O(nlogn) n insert in max cost of log(n) each is O(nlog(n)) total
    """

    def sort(self):
        array = self.mergesort(self.listToArray())

        sorted_tree = AVLTreeList()
        for i in range(len(array)):
            sorted_tree.insert(i,array[i])
        return sorted_tree
    """permute the info values of the list 

    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    @Complexity;
    """

    def permutation(self):
        array = self.listToArray()
        for i in range(len(array)):
            j = round(random.random()*(len(array)-1))
            temp = array[i]
            array[i] = array[j]
            array[j] = temp

        def buildTreeRec(list,left,right,node):
            if left == right:
                leftnode = AVLNode(list[left])
                leftnode.buildLeaf()
                node.setLeft(leftnode)
                node.getLeft().setParent(node)
                node.setSize(2)
                node.setHeight(1)
                return
            if left>right: return
            leftnode = AVLNode(list[left])
            leftnode.buildLeaf()
            node.setLeft(leftnode)
            rightnode = AVLNode(list[right])
            rightnode.buildLeaf()
            node.setRight(rightnode)
            node.getLeft().setParent(node)
            node.getRight().setParent(node)
            buildTreeRec(list,left+1,right//2,node.getLeft())
            if not (right//2 <= left):
                buildTreeRec(list, right//2 + 1, right-1, node.getRight())
            node.updateSizeHeight()


        rand_tree = AVLTreeList()
        rand_tree.root = AVLNode(array[0])
        buildTreeRec(array,1,len(array)-1,rand_tree.root)
        rand_tree.root.updateSizeHeight()
        rand_tree.size = rand_tree.root.getLeft().getSize() +rand_tree.root.getRight().getSize() +1
        rand_tree.lastItem = self.maximum(rand_tree.root)
        rand_tree.firstItem = self.minimum(rand_tree.root)
        return rand_tree




    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst):
        dif = abs(self.root.getHeight() - lst.getRoot().getHeight())
        if(self.root.getHeight() <= lst.getRoot().getHeight()):
            x = self.lastItem
            self.delete(self.size-1)
            x.setLeft(self.root)
            x.getLeft().setParent(x)
            b = lst.getRoot()
            if (b.getHeight() == self.root.getHeight()):
                b=b.getLeft()
            while b.getHeight() > self.root.getHeight():
                b= b.getLeft()
            x.setRight(b)
            c = b.getParent()
            b.setParent(x)
            c.setLeft(x)
            print(c.getValue(),"c",c.getLeft().getValue())
            x.setParent(c)
            self.fixHeightSizeConcat(x)
            self.root = lst.root
            self.size = lst.size + self.size +1
            self.lastItem = lst.lastItem
            self.fixBalanceFactorDelete(x)
        else:
            x = lst.firstItem
            lst.delete(lst.size-1)
            x.setRight(lst.root)
            x.getRight.setParent(x)
            b = self.getRoot()
            if (b.getHeight() == lst.root.getHeight()):
                b=b.getRight()
            while b.getHeight() > lst.root.getHeight():
                b= b.getRight()
            x.setLeft(b)
            c = b.getParent()
            b.setParent(x)
            c.setRight(x)
            x.setParent(c)
            self.fixHeightSizeConcat(x)
            self.size = lst.size + self.size + 1
            self.lastItem = lst.lastItem
            self.fixBalanceFactorDelete(x)


    def fixHeightSizeConcat(self,node):
        while node != None:
            node.setSize(1+node.getLeft().getSize()+node.getRight().getSize())
            node.setHeight(1+max(node.getLeft().getHeight(),node.getRight().getHeight()))
            node = node.getParent()
    """searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

    def search(self, val):
        lst = self.listToArray()
        for i in range(len(lst)):
            if lst[i] == val:
                return i
        return -1


    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return self.root

    """return the 𝒌th smallest element in T
    
    @rtype = AVLNode
    @returns: 𝒌th smallest element, None if k<1 or k>root.size
    @Complexity: maximum height of the tree actions in AVL meaning O(log(n))"""
    def Select(self,node,k):

        r = node.left.getSize() + 1
        if k == r: return node
        elif k < r: return self.Select(node.getLeft(),k)
        elif k > r: return self.Select(node.getRight(),k-r)

    """return the position in sorted order of a given element 𝒙 in 𝐷
    
    @rtype = int
    @returns: rank of element x
    @Complexity: maximum height of the tree actions in AVL meaning O(log(n))"""
    def Rank(self,node):
        r = node.left.size + 1
        y = node
        while(y.isRealNode()):
            if(y == y.getParent().getRight()):
                r = r + y.getParent().getLeft().getSize() + 1
            y = y.getParent()
        return r
    """return the predecessor of a given node
    
    @rtype = AVLNode
    @returns: predecessor
    @Complexity: maximum height of the tree actions in AVL meaning O(log(n))"""
    def predecessor(self,node):
        if node.getLeft().isRealNode():
            return self.maximum(node.getLeft())
        y = node.getParent()
        while(y.isRealNode() and node == y.getLeft()):
            node = y
            y = node.getParent
        return y

    """checking for invalid balance factor after insertion and corrects it
    
    @rtype = None
    @returns:None
    @Complexity: O(log(n)) maximum height of tree action till finding invalid balance factor and O(1) for rotate"""
    def fixBalanceFactorInsert(self,node):
        y = node.getParent()
        while (y is not None):
            bf = y.getBalanceFactor()
            if(abs(bf)>1):
                self.rotateNode(y)
                break
            y = y.getParent()


    """"checking for invalid balance factor after deletion and corrects it
    
    @rtype = None
    @returns:None
    @Complexity: O(log(n)) height of tree actions and O(1) for each rotate"""
    def fixBalanceFactorDelete(self,node):
        y = node
        while (y is not None):
            bf = y.getBalanceFactor()
            if(abs(bf)>1):
                self.rotateNode(y)
            y = y.getParent()

    """rotates an unbalanced node
    
    @rtype:None
    @returns:None
    Complexity: O(1)"""
    def rotateNode(self,node):

        bf = node.getBalanceFactor()
        if (bf == 2):
            son = node.getLeft()
            sonBF = son.getBalanceFactor()
            if(sonBF == 1):
                self.rightRotation(node)
            if (sonBF == 0):
                self.rightRotation(node)
            elif(sonBF == -1):
                self.leftRotation(son)
                self.rightRotation(node)
        elif(bf == -2):
            son = node.getRight()
            sonBf = son.getBalanceFactor()
            if(sonBf == -1):
                self.leftRotation(node)
            if (sonBf == 0):
                self.leftRotation(node)
            elif(sonBf == 1):
                self.rightRotation(son)
                self.leftRotation(node)


    """fixes size and heights of path to the root after insertion
    
    @rtype:None
    @return:None
    @complexity: O(log(n)) path from node to root is max tree height"""
    def fixHeightSizeInsert(self,node):
        y = node
        while(y is not None):
            y.setHeight(max(y.getRight().getHeight(),y.getLeft().getHeight())+1)
            y.setSize(y.getLeft().getSize()+y.getRight().getSize()+1)
            y= y.getParent()

    """fixes size and heights of path to the root after deletion

    @rtype:None
    @return:None
    @complexity: O(log(n)) path from node to root is max tree height"""
    def fixHeightSizeDelete(self,node):
        y = node
        while(y is not None):
            y.setHeight(max(y.getRight().getHeight(),y.getLeft().getHeight())+1)
            y.setSize(y.getLeft().getSize()+y.getRight().getSize()+1)
            y= y.getParent()

    """rotates the node to the right side maintaining size and height
    
    @rtype:None
    @return:None
    @complexity:O(1)"""
    def rightRotation(self,B):
        A = B.getLeft()
        B.setLeft(A.getRight())
        B.getLeft().setParent(B)
        A.setRight(B)
        A.setParent(B.getParent())
        if(B.getParent() == None): self.root = A
        elif(B.getParent().getRight() == B): A.getParent().setRight(A)
        else: A.getParent().setLeft(A)
        B.setParent(A)
        B.updateSizeHeight()
        A.updateSizeHeight()

    """rotates the node to the left side maintaining size and height

    @rtype:None
    @return:None
    @complexity:O(1)"""
    def leftRotation(self,B):
        A = B.getRight()
        B.setRight(A.getLeft())
        B.getRight().setParent(B)
        A.setLeft(B)
        A.setParent(B.getParent())
        if (B.getParent() == None):
            self.root = A
        elif (B.getParent().getRight() == B):
            A.getParent().setRight(A)
        else:
            A.getParent().setLeft(A)
        B.setParent(A)
        B.updateSizeHeight()
        A.updateSizeHeight()


    """"returns the maximum/minimum element (by tree defenition) starting from specific node 
    
    @rtype = AVLNode
    @returns: maximum/minimum element
    complexity: O(log(n)) path from node to max in subtree is max tree height"""
    def maximum(self,node):
        y = node
        while(y.isRealNode()):
            node = y
            y=y.getRight()
        return node

    def minimum(self,node):
        y = node
        while(y.isRealNode()):
            node = y
            y=y.getLeft()
        return node
    """return the root Node
    
    @rtype:AVL Node
    @returns: the root"""
    def getRoot(self):
        return self.root

    """return the tree size

    @rtype:int
    @returns:tree size"""
    """def getSize(self):
        return self.size"""


    def merge(self,A, B):
        """ merging two lists into a sorted list
            A and B must be sorted! """
        n = len(A)
        m = len(B)
        C = [None for i in range(n + m)]

        a = 0;
        b = 0;
        c = 0
        while a < n and b < m:  # more element in both A and B
            if A[a] < B[b]:
                C[c] = A[a]
                a += 1
            else:
                C[c] = B[b]
                b += 1
            c += 1

        C[c:] = A[a:] + B[b:]  # append remaining elements (one of those is empty)

        return C

    def mergesort(self,lst):
        """ recursive mergesort """
        n = len(lst)
        if n <= 1:
            return lst
        else:  # two recursive calls, then merge
            return self.merge(self.mergesort(lst[0:n // 2]),self.mergesort(lst[n // 2:n]))

    def printt(self):
        out = ""
        for row in self.printree(self.root):  # need printree.py file
            out = out + row + "\n"
        print(out)

    def printree(self, t, bykey=True):
        # for row in trepr(t, bykey):
        #        print(row)
        return self.trepr(t, False)

    def trepr(self, t, bykey=False):
        if t == None:
            return ["#"]

        thistr = str(t.key) if bykey else str(t.getValue())

        return self.conc(self.trepr(t.left, bykey), thistr, self.trepr(t.right, bykey))

    def conc(self, left, root, right):

        lwid = len(left[-1])
        rwid = len(right[-1])
        rootwid = len(root)

        result = [(lwid + 1) * " " + root + (rwid + 1) * " "]

        ls = self.leftspace(left[0])
        rs = self.rightspace(right[0])
        result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid *
                      " " + "\\" + rs * "_" + (rwid - rs) * " ")

        for i in range(max(len(left), len(right))):
            row = ""
            if i < len(left):
                row += left[i]
            else:
                row += lwid * " "

            row += (rootwid + 2) * " "

            if i < len(right):
                row += right[i]
            else:
                row += rwid * " "

            result.append(row)

        return result

    def leftspace(self, row):
        # row is the first row of a left node
        # returns the index of where the second whitespace starts
        i = len(row) - 1
        while row[i] == " ":
            i -= 1
        return i + 1

    def rightspace(self, row):
        # row is the first row of a right node
        # returns the index of where the first whitespace ends
        i = 0
        while row[i] == " ":
            i += 1
        return i

    def append(self, val):
        self.insert(self.length(), val)

    def getTreeHeight(self):
        return self.root.getHeight()
tree = AVLTreeList()

tree.insert(0,4)
tree.insert(1,10)
tree.insert(1,7)
tree.insert(3,12)
tree.insert(0,2)
tree.insert(2,5)
tree.insert(4,8)
print("root",tree.getRoot().getValue(),"root height,size:",tree.getRoot().getHeight(),tree.length())
print("root left son",tree.getRoot().getLeft().getValue(),"  height,size ",tree.getRoot().getLeft().getHeight(),tree.getRoot().getLeft().getSize())
print("root right son",tree.getRoot().getRight().getValue(),"  height,size ",tree.getRoot().getRight().getHeight(),tree.getRoot().getRight().getSize())
print("root leftleft son",tree.getRoot().getLeft().getLeft().getValue(),"  height,size ",tree.getRoot().getLeft().getLeft().getHeight(),tree.getRoot().getLeft().getLeft().getSize())
print("root leftright son",tree.getRoot().getLeft().getRight().getValue()," height,size ",tree.getRoot().getLeft().getRight().getHeight(),tree.getRoot().getLeft().getRight().getSize())
print("root rightleft son",tree.getRoot().getRight().getLeft().getValue(),"  height,size ",tree.getRoot().getRight().getLeft().getHeight(),tree.getRoot().getRight().getLeft().getSize())
print("root rightright son",tree.getRoot().getRight().getRight().getValue()," height,size ",tree.getRoot().getRight().getRight().getHeight(),tree.getRoot().getRight().getRight().getSize())

tree.delete(3)
print(tree.search("12"))
print(tree.listToArray())
print(tree.length())
print("root",tree.getRoot().getValue(),"root height,size:",tree.getRoot().getHeight(),tree.length())
print("root left son",tree.getRoot().getLeft().getValue(),"  height,size ",tree.getRoot().getLeft().getHeight(),tree.getRoot().getLeft().getSize())
print("root right son",tree.getRoot().getRight().getValue(),"  height,size ",tree.getRoot().getRight().getHeight(),tree.getRoot().getRight().getSize())
print("root leftleft son",tree.getRoot().getLeft().getLeft().getValue(),"  height,size ",tree.getRoot().getLeft().getLeft().getHeight(),tree.getRoot().getLeft().getLeft().getSize())
print("root leftright son",tree.getRoot().getLeft().getRight().getValue()," height,size ",tree.getRoot().getLeft().getRight().getHeight(),tree.getRoot().getLeft().getRight().getSize())
print("root rightleft son",tree.getRoot().getRight().getLeft().getValue(),"  height,size ",tree.getRoot().getRight().getLeft().getHeight(),tree.getRoot().getRight().getLeft().getSize())
print("root rightright son",tree.getRoot().getRight().getRight().getValue()," height,size ",tree.getRoot().getRight().getRight().getHeight(),tree.getRoot().getRight().getRight().getSize())

print("min value:", tree.first().getValue()," max value:",tree.last().getValue())


tree2 = AVLTreeList()

tree2.insert(0,50)
tree2.insert(1,100)
tree2.insert(1,70)
tree2.insert(0,20)
tree2.insert(4,160)

print(tree2.listToArray())
print(tree2.length())
print("root",tree2.getRoot().getValue(),"root height,size:",tree2.getRoot().getHeight(),tree.length())
print("root left son",tree2.getRoot().getLeft().getValue(),"  root left son height,size ",tree2.getRoot().getLeft().getHeight(),tree2.getRoot().getLeft().getSize())
print("root right son",tree2.getRoot().getRight().getValue(),"  root right son height,size ",tree2.getRoot().getRight().getHeight(),tree2.getRoot().getRight().getSize())
print("root leftleft son",tree2.getRoot().getLeft().getLeft().getValue())
print("root leftright son",tree2.getRoot().getLeft().getRight().getValue())
print("root rightleft son",tree2.getRoot().getRight().getLeft().getValue())
print("root rightright son",tree2.getRoot().getRight().getRight().getValue())
print("min value:", tree2.first().getValue()," max value:",tree.last().getValue())

tree.concat(tree2)

print(tree.listToArray())
print(tree.length())
print("root",tree.getRoot().getValue(),"root height,size:",tree.getRoot().getHeight(),tree.length())
print("root left son",tree.getRoot().getLeft().getValue(),"  root left son height,size ",tree.getRoot().getLeft().getHeight(),tree.getRoot().getLeft().getSize())
print("root right son",tree.getRoot().getRight().getValue(),"  root right son height,size ",tree.getRoot().getRight().getHeight(),tree.getRoot().getRight().getSize())
print("root leftleft son",tree.getRoot().getLeft().getLeft().getValue(),"  root right son height,size ",tree.getRoot().getLeft().getRight().getHeight(),tree.getRoot().getRight().getSize())
print("root leftright son",tree.getRoot().getLeft().getRight().getValue(),"  root right son height,size ",tree.getRoot().getLeft().getRight().getHeight(),tree.getRoot().getRight().getSize())
print("root rightleft son",tree.getRoot().getRight().getLeft().getValue(),"  root right son height,size ",tree.getRoot().getRight().getLeft().getHeight(),tree.getRoot().getRight().getSize())
print("root rightright son",tree.getRoot().getRight().getRight().getValue(),"  root right son height,size ",tree.getRoot().getRight().getRight().getHeight(),tree.getRoot().getRight().getSize())

print("root leftleftleft son",tree.getRoot().getLeft().getLeft().getLeft().getValue())
print("root leftleftright son",tree.getRoot().getLeft().getLeft().getRight().getValue())
print("root leftrightleft son",tree.getRoot().getLeft().getRight().getLeft().getValue())
print("root left-right-right son",tree.getRoot().getLeft().getRight().getRight().getValue())
print("root right-right-right son",tree.getRoot().getRight().getRight().getRight().getValue())
print("min value:", tree.first().getValue()," max value:",tree.last().getValue())
