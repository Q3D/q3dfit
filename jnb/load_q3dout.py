import numpy as np
import sys
sys.path.append("../")
from q3dfit.q3dout import q3dout
from q3dfit.q3df import q3df

s_out = '../jnb/MIRI-ETC-sim/miritest_0006_0012.npy'
struct = np.load(s_out, allow_pickle=True)[()]
q3dout_ij = q3dout(5,11, struct)

# s_out = '../jnb/22128896/22128896_0001_0001.npy'
# struct = np.load(s_out, allow_pickle=True)[()]
# q3dout_ij = q3dout(0,0, struct)

q3dout_ij.get_cont_props()
# q3dout_ij.plot_cont(decompose_qso_fit=True)
q3dout_ij.plot_cont(decompose_qso_fit=True)
breakpoint()

q3dout_ij.plot_lin()



