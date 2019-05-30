from matplotlib import pyplot as plt
import numpy as np

#Define the class for a node of the 'Tree' class. It has to have coordinates of
#the particles it contains, the (x,y) of the bottom left corner, which node
#is its parent and which are its children, what is the zero'th mass moment of
#the node, if it is a leaf node or not and finally the size (width and height).
class Node(object):
    
    def __init__(self,xbl,ybl,size,coordinates,parent):
        self.xbl = xbl
        self.ybl = ybl
        self.size = size
        self.coordinates = coordinates
        self.children = []
        self.parent = parent
        self.leaf = True
        self.m0 = 0
    
    #Checks if a point with the given coordinates is contained in this node.
    def containspt(self,xi,yi):
        if (xi >= self.xbl and xi <= self.xbl + self.size and yi >= self.ybl 
            and yi <= self.ybl + self.size):
            return True
        else:
            return False

#Class to define a tree. The maximum amount of particles inside each node, the
#coordinates of all particles and the mass of each particle should be given.
class Tree(object):
    
    def __init__(self,threshold,coordinates,mass):
        self.threshold = threshold
        self.coordinates = coordinates
        self.root = Node(0,0,150,coordinates,None)
        self.mass = mass
    
    #To build the tree we use the recursive algorithm 'addnodes'
    def build(self):
        addnodes(self.root, self.threshold)
    
    #To plot the tree we get the list of subnodes of the root node, and
    #add a patch for each of these.
    def plottree(self,xlim,ylim):
        subnodes = givelistofsubnodes(self.root)
        c = self.coordinates
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1,1,1)
        plt.title('Quadtree for Colliding.hdf5')
        plt.scatter(c[:,0],c[:,1],s=5)
        for n in subnodes:
            ax.add_patch(plt.Rectangle((n.xbl,n.ybl),n.size,n.size,fill=False))
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('QT')
        plt.clf()
    
    #To calculate the m0's we use this recursive function.
    def calcm0s(self):
        calcm0(self.root,self.mass)
    
    #To find m0 of the nodes that contain the particle with the given
    #coordinates.
    def givem0i(self,xi,yi):
        #Initialize the node we are in now and if it is a leaf.
        isleaf = self.root.leaf
        nnow = self.root
        i = 1
        #Each iteration while it is not a leaf, check in its children if
        #one of them contains the particle with given coordinates.
        while isleaf==False:
            #Counter to 
            i += 1
            for n in nnow.children:
                #If the child contains is, update nnow and isleaf.
                if n.containspt(xi,yi):
                    nnow = n
                    isleaf = n.leaf
        #Initialize array with m0-values.
        m0is = np.zeros(i)
        #Go back up the tree and add the m0-values of the nodes to the array.
        for j in range(i):
            m0is[j] = nnow.m0
            nnow = nnow.parent
        return m0is
            
#Recursive function to build a quadtree.
def addnodes(node, threshold):
    #We do not need to subdivide the node if it has less coordinates than the
    #threshold.
    if len(node.coordinates)<=threshold:
        return
    else:
        #Node is not a leaf anymore.
        node.leaf = False
        #Initialize names for quantities we are going to use
        c = node.coordinates
        s = node.size
        s2 = s/2.
        xbl, ybl = node.xbl, node.ybl
        
        #Create the 4 child-nodes and initialize their info. Per child
        #call to this algorithm to subdivide those if needed.
        mask = (c[:,0]<=xbl+s2) & (c[:,1]<=ybl+s2)
        c1 = c[mask]
        node1 = Node(xbl,ybl,s2,c1,node)
        addnodes(node1,threshold)
        
        mask = (c[:,0]<=xbl+s2) & (c[:,1]>=ybl+s2)
        c1 = c[mask]
        node2 = Node(xbl,ybl+s2,s2,c1,node)
        addnodes(node2,threshold)

        mask = (c[:,0]>=xbl+s2) & (c[:,1]>=ybl+s2)
        c1 = c[mask]
        node3 = Node(xbl+s2,ybl+s2,s2,c1,node)
        addnodes(node3,threshold)
        
        mask = (c[:,0]>=xbl+s2) & (c[:,1]<=ybl+s2)
        c1 = c[mask]
        node4 = Node(xbl+s2,ybl,s2,c1,node)
        addnodes(node4,threshold)
        
        #Add the four new nodes as children of the current node.
        node.children = [node1, node2, node3, node4]

#Function to give a list of all nodes that are in some way a subnode of the
#argument.
def givelistofsubnodes(node):
    #If the list of children is empty, we return a list containing the node.
    if not node.children:
        return [node]
    #If the list of children is not empty, we need to append all lists with
    #nodes that are subnodes of said children.
    else:
        subnodes = []
        for c in node.children:
            subnodes += givelistofsubnodes(c)
        return subnodes

#Function to find the 0th order mass moment for each node and store it in that
#nodes information
def calcm0(node,pmass):
    #Only if it is a leaf we can cound the number of particles and multiply
    #with the given particle-mass
    if node.leaf:
        m = len(node.coordinates) * pmass
        node.m0 = m
        return m
    #Else we add the m0's of all the subnodes to get m0 of the current one.
    else:
        s = 0
        for n in node.children:
            s += calcm0(n,pmass)
        node.m0 = s
        return s

