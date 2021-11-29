
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
            if 'global_ext_model' in initdat['argscontfit'] or ('args_questfit' in initdat['argscontfit'] and 'global_ext_model' in initdat['argscontfit']['args_questfit']):
               str_global_ext = list(comp_best_fit.keys())[-2]
               str_global_ice = list(comp_best_fit.keys())[-1]
               if len(comp_best_fit[str_global_ext].shape) > 1:  # global_ext is a multi-dimensional array
                  comp_best_fit[str_global_ext] = comp_best_fit[str_global_ext] [:,0,0]
               if len(comp_best_fit[str_global_ice].shape) > 1:  # global_ice is a multi-dimensional array
                  comp_best_fit[str_global_ice] = comp_best_fit[str_global_ice] [:,0,0]
               for i, el in enumerate(comp_best_fit):
                  if (el != str_global_ext) and (el != str_global_ice):
                    if len(comp_best_fit[el].shape) > 1:              # component is a multi-dimensional array
                      comp_best_fit[el] = comp_best_fit[el] [:,0,0]
                    ax1.plot(MIRgdlambda_temp, comp_best_fit[el]*comp_best_fit[str_global_ext]*comp_best_fit[str_global_ice], label=el,linestyle='--',alpha=0.5)
            else:

              for i, el in enumerate(comp_best_fit):
                if len(comp_best_fit[el].shape) > 1:
                  comp_best_fit[el] = comp_best_fit[el] [:,0,0]

                if not ('_ext' in el or '_abs' in el):
                    spec_i = comp_best_fit[el]
                    label_i = el
                    if el+'_ext' in comp_best_fit.keys():
                        spec_i = spec_i*comp_best_fit[el+'_ext']
                    if el+'_abs' in comp_best_fit.keys():
                        spec_i = spec_i*comp_best_fit[el+'_abs']
                    ax1.plot(MIRgdlambda_temp, spec_i, label=label_i,linestyle='--',alpha=0.5)



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
