
import numpy as np
import copy



'''
=========================================

 MyVar

=========================================
'''
class Var2d():

    def __init__(self, val):
        self.val = np.array(copy.deepcopy(val), ndmin=2)
        self.rows, self.cols = np.shape(self.val)
        self.grad = np.zeros_like(self.val)
        self.creator = None

    def Set(self, val):
        self.val = np.array(val, dtype=float, ndmin=2)
        #self.val[:,:] = val[:,:]

    def Evaluate(self):
        if self.creator!=None:
            self.val = self.creator.Evaluate()
        return self.val

    def SetCreator(self, op):
        self.creator = op

    def ZeroGrad(self):
        self.grad = np.zeros_like(self.val)
        if self.creator!=None:
            self.creator.ZeroGrad()

    def Backward(self, s=None):
        if s is None:
            s = np.ones_like(self.val)
        self.grad = self.grad + s
        if self.creator!=None:
            self.creator.Backward(s)

    def __call__(self):
        return self.val

    def __add__(self, b):
        return MatPlus(self, b)

    def __matmul__(self, b):
        return MatMul(self, b)

    def __rmatmul__(self, a):
        return MatMul(a, self)

'''
=========================================

 Wrapper Functions

=========================================
'''
# Binary functions ====================
def MatMul(a, b):
    op = MyMatMul([a, b])
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c

def MatPlus(a, b):
    op = MyMatPlus([a,b])
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c

# Unary functions =====================
# Activation functions
def MatLogistic(a):
    op = MyMatLogistic(a)
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c

def MatIdentity(a):
    op = MyMatIdentity(a)
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c

def MatReLU(a):
    op = MyMatReLU(a)
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c

# Reducing functions
def MatSum(a):
    op = MyMatSum(a)
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c

def MatBatchMean(a):
    op = MyMatBatchMean(a)
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c


# Functions with additional parameters ========
def MatSquaredError(a, target):
    op = MyMatSquaredError(a, target)
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c

def MatCE(a, target):
    op = MyMatCE(a, target)
    c = Var2d(op.Evaluate())
    c.SetCreator(op)
    return c

def MatMeanCE(a, target):
    op1 = MyMatCE(a, target)
    c1 = Var2d(op1.Evaluate())
    c1.SetCreator(op1)
    op2 = MyMatBatchMean(c1)
    c2 = Var2d(op2.Evaluate())
    c2.SetCreator(op2)
    return c2

def MatMSE(a, target):
    return MatBatchMean( MatSquaredError(a, target) )


'''
=========================================

 MyOp

=========================================
'''
class MyOp():

    def __init__(self):
        self.args = []

    def Evaluate(self):
        raise NotImplementedError

    def ZeroGrad(self):
        for a in self.args:
            a.ZeroGrad()

    def Backward(self, s=1.):
        raise NotImplementedError


'''
=========================================

 Operation Implementations

=========================================
'''
class MyMatMul(MyOp):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def Evaluate(self):
        val = self.args[0].Evaluate() @ self.args[1].Evaluate()
        return val

    def Backward(self, s=None):
        self.args[0].Backward(s@self.args[1].val.T)
        self.args[1].Backward(self.args[0].val.T@s)


class MyMatPlus(MyOp):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def Evaluate(self):
        val = self.args[0].Evaluate() + self.args[1].Evaluate()
        return val

    def Backward(self, s=None):
        #print(s.shape, self.args[0].val.shape)
        self.args[0].Backward(s*np.ones_like(self.args[0].val))
        self.args[1].Backward(s*np.ones_like(self.args[1].val))



# Activation Functions

class MyMatLogistic(MyOp):
    def __init__(self, arg):
        super().__init__()
        self.args = [arg]

    def Evaluate(self):
        val = 1./( 1. + np.exp(-self.args[0].Evaluate()) )
        self.deriv = val * (1.-val)
        return val

    def Backward(self, s=1.):
        deriv = s * self.deriv
        self.args[0].Backward(deriv)

class MyMatIdentity(MyOp):
    def __init__(self, arg):
        super().__init__()
        self.args = [arg]

    def Evaluate(self):
        val = self.args[0].Evaluate()
        return val

    def Backward(self, s=1.):
        self.args[0].Backward(s)

class MyMatReLU(MyOp):
    def __init__(self, arg):
        super().__init__()
        self.args = [arg]

    def Evaluate(self):
        val = np.clip(self.args[0].Evaluate(), 0, None)
        return val

    def Backward(self, s=1.):
        val = np.ceil( np.clip(self.args[0].val, 0, 1) )
        self.args[0].Backward(s*val)



# Reducing functions

class MyMatSum(MyOp):
    def __init__(self, arg):
        super().__init__()
        self.args = [arg]

    def Evaluate(self):
        val = np.sum(self.args[0].Evaluate(), axis=1, keepdims=True)
        return val

    def Backward(self, s=1.):
        self.args[0].Backward(s*np.ones_like(self.args[0].val))


class MyMatBatchMean(MyOp):
    '''
     Computes the mean over batches. If the input is PxN (ie. P samples)
     then the output is 1xN.
    '''
    def __init__(self, arg):
        super().__init__()
        self.args = [arg]

    def Evaluate(self):
        P = self.args[0].rows
        val = np.sum(self.args[0].Evaluate(), axis=0, keepdims=True)/P
        return val

    def Backward(self, s=1.):
        self.args[0].Backward(s*np.ones_like(self.args[0].val)/self.args[0].rows)



# Functions with additional parameters ========

class MyMatSquaredError(MyOp):
    '''
     MyMatSquaredError(x, target)

     Computes the squared error for each sample (each row).
    '''
    def __init__(self, arg, target):
        super().__init__()
        self.args = [arg]
        self.target = target

    def SetTarget(self, target):
        self.target = target

    def Evaluate(self):
        self.deriv = self.args[0].Evaluate() - self.target
        val = 0.5 * np.sum(self.deriv**2, axis=1)
        return val

    def Backward(self, s=None):
        self.args[0].Backward(s@self.deriv)



class MyMatCE(MyOp):
    '''
     Computes cross entropy (CE) for each sample (and sums the result if
     there are multiple outputs. For example, if the input is PxN, then
     the output is Px1.
    '''
    def __init__(self, arg, target):
        super().__init__()
        self.args = [arg]
        self.target = target

    def SetTarget(self, target):
        self.target = target

    def Evaluate(self):
        P, N = self.args[0].rows, self.args[0].cols
        aval = self.args[0].Evaluate()
        val = -np.sum(self.target*np.log(aval) + (1-self.target)*np.log(1.-aval), axis=1, keepdims=True)
        self.deriv = aval - self.target
        return val

    def Backward(self, s=1.):
        '''
         deriv = mm.Backward(s=1.)

         Multiplies s by the gradient of this operator.
        '''
        deriv = self.deriv/self.args[0].val/(1.-self.args[0].val)
        self.args[0].Backward(s*deriv)
