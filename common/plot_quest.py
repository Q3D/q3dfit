
import numpy as np
from matplotlib import pyplot as plt


def plot_quest(MIRgdlambda, MIRgdflux, MIRcontinuum, ct_coeff, initdat, templ_mask=[], lines=[], linespec=[]):
    comp_best_fit = ct_coeff['comp_best_fit']

    if 'plotMIR' in initdat.keys(): # Test plot - To do: move this to the correct place in q3da
      if initdat['plotMIR']:
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(6, 7))
        gs = fig.add_gridspec(4,1)
        ax1 = fig.add_subplot(gs[:3, :])

        ax1.plot(MIRgdlambda, MIRgdflux,color='black')
        ax1.plot(MIRgdlambda, MIRcontinuum)

        if len(templ_mask)>0:
          MIRgdlambda_temp = MIRgdlambda[templ_mask]
        else:
          MIRgdlambda_temp = MIRgdlambda

        if len(lines)>0:
          for line_i in lines:
            ax1.axvline(line_i, color='grey', linestyle='--', alpha=0.7, zorder=0)
            #ax1.axvspan(line_i-max(initdat['siglim_gas']), line_i+max(initdat['siglim_gas']))
          ax1.plot(MIRgdlambda, linespec, color='r', linestyle='-', alpha=0.7, linewidth=1.5)

        if 'argscontfit' in initdat:
            if 'global_ext_model' in initdat['argscontfit']:
               for i in np.arange(0,len(comp_best_fit.keys())-2,1):
                  if len(comp_best_fit[list(comp_best_fit.keys())[i]].shape) > 1:
                    comp_best_fit[list(comp_best_fit.keys())[i]] = comp_best_fit[list(comp_best_fit.keys())[i]] [:,0,0]
                  if len(comp_best_fit[list(comp_best_fit.keys())[-2]].shape) > 1:
                    comp_best_fit[list(comp_best_fit.keys())[-2]] = comp_best_fit[list(comp_best_fit.keys())[-2]] [:,0,0]
                  if len(comp_best_fit[list(comp_best_fit.keys())[-1]].shape) > 1:
                    comp_best_fit[list(comp_best_fit.keys())[-1]] = comp_best_fit[list(comp_best_fit.keys())[-1]] [:,0,0]
                  ax1.plot(MIRgdlambda_temp,comp_best_fit[list(comp_best_fit.keys())[i]]*comp_best_fit[list(comp_best_fit.keys())[-2]]*comp_best_fit[list(comp_best_fit.keys())[-1]],label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)

            else:
               for i in np.arange(0,len(comp_best_fit.keys()),3):
                  if len(comp_best_fit[list(comp_best_fit.keys())[i]].shape) > 1:
                    comp_best_fit[list(comp_best_fit.keys())[i]] = comp_best_fit[list(comp_best_fit.keys())[i]] [:,0,0]
                  if len(comp_best_fit[list(comp_best_fit.keys())[i+1]].shape) > 1:
                    comp_best_fit[list(comp_best_fit.keys())[i+1]] = comp_best_fit[list(comp_best_fit.keys())[i+1]] [:,0,0]
                  if len(comp_best_fit[list(comp_best_fit.keys())[i+2]].shape) > 1:
                    comp_best_fit[list(comp_best_fit.keys())[i+2]] = comp_best_fit[list(comp_best_fit.keys())[i+2]] [:,0,0]
                  ax1.plot(MIRgdlambda_temp,comp_best_fit[list(comp_best_fit.keys())[i]]*comp_best_fit[list(comp_best_fit.keys())[i+1]]*comp_best_fit[list(comp_best_fit.keys())[i+2]],label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
        else:
            print('argscontfit  missing in the init dict \n --> Not plotting MIR fitting results correctly.')

        ax1.legend(ncol=2)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xticklabels([])
        ax1.set_ylim(1e-5,1e2)

        ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
        ax2.plot(MIRgdlambda,MIRgdflux/MIRcontinuum,color='black')
        ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
        ax2.set_ylabel('Data/Model')
        ax2.set_xlabel('Wavelength [micron]')
        gs.update(wspace=0.0, hspace=0.05)

        plt.savefig('../test/test_questfit/'+initdat['label'])
        plt.show()