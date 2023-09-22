from ..apptools import *
from .nmaoptions import *
from . import nmaoptions

__all__ = ['prody_simplemd']

def prody_simplemd():
    smd = prody.SimpleMD()
    smd.run(pdbname, ph=7, forcefield, temp=303.5,
            heat_steps, prod_steps, platform,
            extend=False, extend_steps=0, **kwargs):
