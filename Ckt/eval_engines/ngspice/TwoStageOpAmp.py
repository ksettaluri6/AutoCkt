""" 
# Fully Differential OTA Example 

Highlights the capacity to use `Diff` signals and `Pair`s of instances 
for differential circuits. 

"""

import sys
from copy import deepcopy
import hdl21 as h
import hdl21.sim as hs


""" 
Create a small "PDK" consisting of an externally-defined Nmos and Pmos transistor. 
Real versions will have some more parameters; these just have multiplier "m". 
"""


@h.paramclass
class MosParams:
    m = h.Param(dtype=int, desc="Transistor Multiplier")


nmos = h.ExternalModule(
    name="nmos",
    desc="Nmos Transistor (Multiplier Param Only!)",
    port_list=deepcopy(h.Mos.port_list),
    paramtype=MosParams,
)
pmos = h.ExternalModule(
    name="pmos",
    desc="Pmos Transistor (Multiplier Param Only!)",
    port_list=deepcopy(h.Mos.port_list),
    paramtype=MosParams,
)

@h.paramclass
class OpAmpParams:
    """Parameter class"""
    wp1 = h.Param(dtype=int, desc="Width of PMOS mp1", default=10)
    wp2 = h.Param(dtype=int, desc="Width of PMOS mp2", default=10)
    wp3 = h.Param(dtype=int, desc="Width of PMOS mp3", default=4)
    wn1 = h.Param(dtype=int, desc="Width of NMOS mn1", default=38)
    wn2 = h.Param(dtype=int, desc="Width of NMOS mn2", default=38)
    wn3 = h.Param(dtype=int, desc="Width of NMOS mn3", default=9)
    wn4 = h.Param(dtype=int, desc="Width of NMOS mn4", default=20)
    wn5 = h.Param(dtype=int, desc="Width of NMOS mn5", default=60)
    VDD = h.Param(dtype=float, desc="VDD voltage", default=1)
    CL = h.Param(dtype=float, desc="CL capacitance", default=1)
    Cc = h.Param(dtype=float, desc="Cc capacitance", default=1)
    ibias = h.Param(dtype=float, desc="ibias current", default=1)


@h.generator
def OpAmp(p: OpAmpParams) -> h.Module:
    """# Two stage OpAmp """

    @h.module
    class DiffOta:
        # IO Interface
        VDD, VSS = 2 * h.Input()
        inp = h.Diff(desc="Differential Input", port=True, role=h.Diff.Roles.SINK)
        out = h.Diff(desc="Differential Output", port=True, role=h.Diff.Roles.SOURCE)

        # Internal Signals
        net1, net2, net3, net4, net5, net6, net7 = h.Signals(7)
        # cm = h.Signal()

        # Input Stage & CMFB Bias
        mp1 = pmos(m=p.wp1)(d=net4, g=net4, s=VDD, b=VDD)
        mp2 = pmos(m=p.wp2)(d=net5, g=net4, s=VDD, b=VDD)
        mn1 = pmos(m=p.wn1)(d=net4, g=net1, s=net3, b=net3)
        mn2 = pmos(m=p.wn2)(d=net5, g=net1, s=net3, b=net3)
        mn3 = pmos(m=p.wn3)(d=net3, g=net7, s=VSS, b=VSS)

        # Output Stage
        mp3 = pmos(m=p.wp3)(d=net6, g=net5, s=VDD, b=VDD)
        mn5 = nmos(m=p.wn5)(d = net6, g = net7, s = VSS, b = VSS)
        CL = h.Cap(c=p.CL)(p = net6, n = VSS)

        # Biasing
        mn4 = nmos(m=p.wn4)(d = net7, g = net7, s = VSS, b = VSS)
        ibias = h.Isrc(dc = p.ibias)(p = net7, n = VDD)

        net1=inp.p
        net2=inp.n
        out.p=net6
        out.n=VSS

        # xndiode = nmos(m=1)(d=ibias, g=ibias, s=VSS, b=VSS)
        # xnsrc = nmos(m=1)(d=pbias, g=ibias, s=VSS, b=VSS)
        # xpdiode = pmos(m=6)(d=pbias, g=pbias, s=VDD, b=VDD)

        # Compensation Network
        Cc = h.Cap(c = p.Cc)(p = net5, n = net6)

    return DiffOta


@h.module
class CapCell:
    """# Compensation Capacitor Cell"""

    p, n, VDD, VSS = 4 * h.Port()
    # FIXME: internal content! Using tech-specific `ExternalModule`s


@h.module
class ResCell:
    """# Compensation Resistor Cell"""

    p, n, sub = 3 * h.Port()
    # FIXME: internal content! Using tech-specific `ExternalModule`s


@h.module
class Compensation:
    """# Single Ended RC Compensation Network"""

    a, b, VDD, VSS = 4 * h.Port()
    r = ResCell(p=a, sub=VDD)
    c = CapCell(p=r.n, n=b, VDD=VDD, VSS=VSS)


@hs.sim
class MosDcopSim:
    """# Mos Dc Operating Point Simulation Input"""

    @h.module
    class Tb:
        """# Basic Mos Testbench"""

        VSS = h.Port()  # The testbench interface: sole port VSS
        vdc = h.Vdc(dc=1.2)(n=VSS)  # A DC voltage source
        dcin = h.Diff()
        sigp = h.Vdc(dc=0.6)(n=VSS)
        sign = h.Vdc(dc=0.55)(n=VSS)
        dcin.p=sigp.p
        dcin.n=sign.n
        inst=OpAmp()(VDD=vdc.p, VSS=VSS, inp=dcin)

    # Simulation Stimulus
    op = hs.Op()
    mod = hs.Include("spice_models/45nm_bulk.txt")


def main():
    h.netlist(OpAmp(), sys.stdout)


if __name__ == "__main__":
    main()
