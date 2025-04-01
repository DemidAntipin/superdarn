import sys
import numpy as np
import matplotlib.pyplot as pp

f=sys.stdin
for s in f:
 d=np.array(s.split()).astype(float)
 p=d[3:]
 fig,axs=pp.subplots(5,1,constrained_layout=True,figsize=(3,6))

 axs[0].set_title('Abs')
 axs[0].plot(p[:-1:3],label='Abs')
 axs[1].set_title('Phase')
 axs[1].plot(p[1:-1:3],label='Phase')
 axs[2].set_title('Re')
 axs[2].plot(p[:-1:3]*np.cos(p[1:-1:3]),label='Re')
 axs[3].set_title('Im')
 axs[3].plot(p[:-1:3]*np.sin(p[1:-1:3]),label='Im')
 axs[4].set_title('Qty')
 axs[4].plot(p[2:-1:3],label='Qty')
# pp.plot(p[1:-1:3])
 pp.legend()
 pp.show()
