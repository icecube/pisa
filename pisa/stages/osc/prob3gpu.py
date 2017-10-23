# authors: T.Arlen, J.Lanfranchi, P.Eller
# date:   March 20, 2016


from __future__ import division

from pisa.stages.osc.prob3 import prob3


__all__ = ['prob3gpu']



class prob3gpu(prob3):

    def __init__(self,**kw):
        super(self.__class__, self).__init__(use_gpu=True,**kw)
