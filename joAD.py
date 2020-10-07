
import numpy as np



'''
=========================================

 MyVar

=========================================
'''
class MyVar():

    def __init__(self, val):
        self.val = val
        self.grad = None
        self.creator = None

    def Set(self, val):
        self.val = val

    def Evaluate(self):
        if self.creator!=None:
            self.val = self.creator.Evaluate()
        return self.val

    def __repr__(self):
        if self.creator==None:
            return str(self.val)
        else:
            return self.creator.__repr__()

    def SetCreator(self, op):
        self.creator = op

    def ZeroGrad(self):
        self.grad = 0.
        if self.creator!=None:
            self.creator.ZeroGrad()

    def Backward(self, s=1.):
        self.grad += s
        if self.creator!=None:
            self.creator.Backward(s)

    def __call__(self):
        return self.val


'''
=========================================

 Wrapper Functions

=========================================
'''
def Mul(a, b):
    op = MyMul([a, b])
    c = MyVar(op.Evaluate())
    c.SetCreator(op)
    return c

def Plus(a, b):
    op = MyPlus([a, b])
    c = MyVar(op.Evaluate())
    c.SetCreator(op)
    return c

def Recip(a):
    op = MyRecip([a])
    c = MyVar(op.Evaluate())
    c.SetCreator(op)
    return c

def Power(a, power=2):
    op = MyPower([a], power=power)
    c = MyVar(op.Evaluate())
    c.SetCreator(op)
    return c

def Log(a):
    op = MyLog([a])
    c = MyVar(op.Evaluate())
    c.SetCreator(op)
    return c




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
class MyLog(MyOp):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def Evaluate(self):
        val = np.log(self.args[0].Evaluate())
        return val

    def __repr__(self):
        return 'log('+self.args[0].__repr__()+')'

    def Backward(self, s=1.):
        '''
         deriv = mm.Backward(s=1.)

         Multiplies s by the gradient of this operator.
        '''
        deriv = 1./self.args[0].val
        self.args[0].Backward(s*deriv)


class MyPower(MyOp):
    def __init__(self, args, power=2):
        super().__init__()
        self.args = args
        self.power = power

    def Evaluate(self):
        val = self.args[0].Evaluate()**self.power
        return val

    def __repr__(self):
        return '('+self.args[0].__repr__()+')**'+str(self.power)

    def Backward(self, s=1.):
        '''
         deriv = mm.Backward(s=1.)

         Multiplies s by the gradient of this operator.
        '''
        deriv = self.power*self.args[0].val**(self.power-1)
        self.args[0].Backward(s*deriv)


class MyPlus(MyOp):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def Evaluate(self):
        val = self.args[0].Evaluate() + self.args[1].Evaluate()
        return val

    def __repr__(self):
        return '('+self.args[0].__repr__()+'+'+str(self.args[1].__repr__())+')'

    def Backward(self, s=1.):
        '''
         deriv = mm.Backward(s=1.)

         Multiplies s by the gradient of this operator.
        '''
        self.args[0].Backward(s)
        self.args[1].Backward(s)


class MyRecip(MyOp):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def Evaluate(self):
        val = 1./self.args[0].Evaluate()
        return val

    def __repr__(self):
        return '(1/'+self.args[0].__repr__()+')'

    def Backward(self, s=1.):
        '''
         deriv = mm.Backward(s=1.)

         Multiplies s by the gradient of this operator.
        '''
        deriv = -1./self.args[0].val**2
        self.args[0].Backward(s*deriv)


class MyMul(MyOp):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def Evaluate(self):
        val = self.args[0].Evaluate() * self.args[1].Evaluate()
        return val

    def __repr__(self):
        return self.args[0].__repr__()+'*'+str(self.args[1].__repr__())

    def Backward(self, s=1.):
        '''
         deriv = mm.Backward(s=1.)

         Multiplies s by the gradient of this operator.
        '''
        x_deriv = self.args[1].val
        y_deriv = self.args[0].val
        self.args[0].Backward(s*x_deriv)
        self.args[1].Backward(s*y_deriv)
