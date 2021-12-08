from inspect import trace
import torch

class MyDecisionGate(torch.nn.Module):
    """[summary]

    Args:
        torch ([type]): [description]
    """
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x
            

class MyCell(torch.nn.Module):
    """[summary]

    Args:
        torch ([type]): [description]
    """
    def __init__(self):
        super(MyCell, self).__init__()
        self.dg = MyDecisionGate()
        self.linear = torch.nn.Linear(4, 4)
        
    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        
        return new_h, new_h
    
my_cell = MyCell()
x = torch.rand(3, 4) # batch, features 
h = torch.rand(3, 4)

# traced_cell = torch.jit.trace(my_cell, (x, h))
scripted_cell = torch.jit.script(my_cell)
print(scripted_cell)
print(type(scripted_cell))
print(scripted_cell.dg.code)
print(scripted_cell.code)

print(scripted_cell(x, h))