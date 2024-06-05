"""
This is used to define a serializable object used for functions assigned to the DerivedParams 

These can be constructed and evaluated symbolically and procedurally. 
In principle, we may even be able to include a filepath to a seriialized Funct object such that the pipeline configs can include definitions for these therein 

Contains
    OPS - an Enum listing recognized operations 
    TrigOps - some definitions for handling some of the trig functions, shared by Vars and Functs 
    Var - a class for representing variables 
    Funct - a series of operations done representing the functions 

Uses - quite simple! 

create some vars and do math on them 

x = Var('x')
y = Var('y')

function = sin(x**2) + 3*cos(y+1)**2 

The object `function` is now callable with keyword arguments passed to the instantiated `Vars`
"""
# from typing import Callable
from pisa.utils import jsons
from enum import Enum

import math
import numpy as np

class OPS(Enum):
    """
    Enumerate different operations so that the Funct class can do math 
    """
    ADD = 0
    MUL = 1
    POW = 2
    SIN = 3
    COS = 4 
    TAN = 5 

    @property
    def state(self):
        return self.serializable_state

    @property
    def serializable_state(self):
        return {"ops":self.value, "kind":"ops"}

    @classmethod 
    def from_state(cls, state):
        return cls(state["ops"])

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.
        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, filename):
        """Instantiate a new Param from a JSON file"""
        state = jsons.from_json(filename=filename)
        return OPS(state["ops"])



class TrigOps:
    """
    These are all used by both the Var and Funct classes, so there's some fun python hierarchy stuff going on instead 
    """
    @property
    def sin(self):
        new_op = Funct(self)
        new_op.add_opp(OPS.SIN, 0.0)
        return new_op

    @property
    def cos(self):
        new_op = Funct(self)
        new_op.add_opp(OPS.COS, 0.0)
        return new_op

    @property
    def tan(self):
        new_op = Funct(self)
        new_op.add_opp(OPS.TAN, 0.0)
        return new_op

class Var(TrigOps):
    """
    A variable

    These are a lot like functions in how they are combined, but instead evaluate simply to one of the keyword arguments passed to Functions
    """
    # the id is used to assign unique names to each variable in the event that the user does not manually specify a name 
    _ids = 0
    def __init__(self, name=None):
        if name is None:
            self._name = "arg"+str(Var._ids)
        else:
            self._name = name
        Var._ids+=1 

    @property
    def state(self):
        return {
            "kind":"var",
            "name": self._name
        }

    @property
    def serializable_state(self):
        return self.state

    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.
        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, filename):
        """Instantiate a new Param from a JSON file"""
        state = jsons.from_json(filename=filename)
        return Var.from_state(state)

    @classmethod 
    def from_state(cls, state):
        return cls(state["name"])

    @property 
    def name(self):
        return self._name

    def __call__(self, **kwargs):
        # NOTE we implicitly down-cast everything to a float/int here! 

        value = kwargs[self._name]
        if type(value)==list:
            raise ValueError("Lists aren't supported. This is probably wrong")
        return value.value.m

    def __add__(self, other):
        new = Funct(self)
        new = new + other
        return new

    def __mul__(self, other):
        new = Funct(self)
        new = new * other
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        new = Funct(self)
        new = new ** other
        return new

class Funct(TrigOps):
    """
    Functions are constructed as a series of operations one to some starting value. 
    The starting value can be whatever - a value, function or variable 
    """

    def __init__(self, first_var):
        self._ops = [(OPS.ADD, first_var)]

    def __call__(self,**kwargs):
        value = 0.0
        for op in self._ops:
            if op[0] == OPS.ADD:
                if isinstance(op[1],(Funct, Var)):
                    value += op[1](**kwargs)
                else:
                    value += op[1]
            elif op[0] == OPS.MUL:
                if isinstance(op[1], (Funct, Var)):
                    value *= op[1](**kwargs)
                else:
                    value *= op[1]
            elif op[0] == OPS.POW:
                if isinstance(op[1], (Funct, Var)):
                    value **= op[1](**kwargs)
                else:
                    value **= op[1]
            elif op[0] == OPS.SIN:
                if isinstance(value, np.ndarray):
                    value = np.sin(value) 
                else:
                    value = math.sin(value) # significantly faster for non-arrays 
            elif op[0] == OPS.COS:
                if isinstance(value, np.ndarray):
                    value = np.cos(value)
                else:
                    value = math.cos(value)
            elif op[0] == OPS.TAN:
                if isinstance(value, np.ndarray):
                    value = np.tan(value)
                else:
                    value = math.tan(value)

        return value

    def add_opp(self, kind:OPS, other):
        self._ops.append((kind, other))

    def __add__(self, other):
        self.add_opp(OPS.ADD, other)
        return self
        
    def __mul__(self, other):
        self.add_opp(OPS.MUL, other)
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        self.add_opp(OPS.POW, other)
        return self

    ############## some functions to handle serializing these objects 
    @property
    def state(self):
        statekind = {}
        statekind["kind"] ="Funct"
        statekind["ops"]=[]
        for entry in self._ops:
            if isinstance(entry[1], (Funct, Var, OPS)):
                sub_state = entry[1].state
            else:
                sub_state = entry[1]

            statekind["ops"].append([entry[0].serializable_state, sub_state])

        return statekind

    @property
    def serializable_state(self):
        return self.state
    

    @classmethod
    def from_state(cls, state):
        new_op = cls(0.0)
        statedict = state["ops"]
        for entry in statedict:
            op = OPS.from_state(entry[0])
            if isinstance(entry[1],  dict):
                if entry[1]["kind"]=="Funct":
                    entry_class = Funct
                elif entry[1]["kind"]=="var":
                    entry_class = Var
                elif entry[1]["kind"]=="ops":
                    entry_class = OPS
                else:
                    raise ValueError("Cannot de-serialzie {}".format(entry[1]["kind"]))
                
                value = entry_class.from_state(entry[1])
            else:
                value = entry[1]
            new_op.add_opp(op, value)
        return new_op


    def to_json(self, filename, **kwargs):
        """Serialize the state to a JSON file that can be instantiated as a new
        object later.
        """
        jsons.to_json(self.serializable_state, filename=filename, **kwargs)

    @classmethod
    def from_json(cls, filename):
        """Instantiate a new Param from a JSON file"""
        state = jsons.from_json(filename=filename)
        return Funct.from_state(state)

# some macros for readability 
def sin(target:Funct):
    return target.sin
def cos(target:Funct):
    return target.cos
def tan(target:Funct):
    return target.tan
