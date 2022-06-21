
import numpy as np
from matplotlib import pyplot as plt
from q3dfit import questfit_readcf


def plot_quest(MIRgdlambda, MIRgdflux, MIRcontinuum, ct_coeff, initdat,
               templ_mask=[], lines=[], linespec=[]):

    comp_best_fit = ct_coeff['comp_best_fit']

    plot_noext = False

    if 'argscontfit' in initdat.keys() and 'plot_decomp' in initdat['argscontfit'].keys(): # Test plot - To do: move this to the correct place in q3da
      if initdat['argscontfit']['plot_decomp']:
        config_file = questfit_readcf.readcf(initdat['argscontfit']['config_file'])
        global_extinction = False
        for key in config_file:
            try:
                if 'global' in config_file[key][3]:
                        global_extinction = True
            except:
                continue

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


        colour_list = ['dodgerblue', 'mediumblue', 'salmon', 'palegreen', 'orange', 'purple', 'forestgreen', 'darkgoldenrod', 'mediumblue', 'magenta', 'plum', 'yellowgreen']

        if global_extinction:
            str_global_ext = list(comp_best_fit.keys())[-2]
            str_global_ice = list(comp_best_fit.keys())[-1]
            # global_ext is a multi-dimensional array
            if len(comp_best_fit[str_global_ext].shape) > 1:
                comp_best_fit[str_global_ext] = comp_best_fit[str_global_ext] [:,0,0]
            # global_ice is a multi-dimensional array
            if len(comp_best_fit[str_global_ice].shape) > 1:
                comp_best_fit[str_global_ice] = comp_best_fit[str_global_ice] [:,0,0]
            count = 0
            for i, el in enumerate(comp_best_fit):
                if (el != str_global_ext) and (el != str_global_ice):
                    if len(comp_best_fit[el].shape) > 1:              # component is a multi-dimensional array
                        comp_best_fit[el] = comp_best_fit[el] [:,0,0]
                    if plot_noext:
                        if count>len(colour_list)-1:
                            ax1.plot(MIRgdlambda_temp, comp_best_fit[el]/comp_best_fit[str_global_ext]/comp_best_fit[str_global_ice], label=el,linestyle='--',alpha=0.5)
                        else:
                            ax1.plot(MIRgdlambda_temp, comp_best_fit[el]/comp_best_fit[str_global_ext]/comp_best_fit[str_global_ice], color=colour_list[count], label=el,linestyle='--',alpha=0.5)
                    else:
                        if count>len(colour_list)-1:
                            ax1.plot(MIRgdlambda_temp, comp_best_fit[el], label=el,linestyle='--',alpha=0.5)
                        else:
                            ax1.plot(MIRgdlambda_temp, comp_best_fit[el], color=colour_list[count], label=el,linestyle='--',alpha=0.5)
                    count += 1

        else:
            count = 0

            for i, el in enumerate(comp_best_fit):
                if len(comp_best_fit[el].shape) > 1:
                  comp_best_fit[el] = comp_best_fit[el] [:,0,0]

                if not ('_ext' in el or '_abs' in el):
                    spec_i = comp_best_fit[el]
                    label_i = el
                    if plot_noext:
                      if el+'_ext' in comp_best_fit.keys():
                          spec_i = spec_i/comp_best_fit[el+'_ext']
                      if el+'_abs' in comp_best_fit.keys():
                          spec_i = spec_i/comp_best_fit[el+'_abs']
                    if count>len(colour_list)-1:
                      ax1.plot(MIRgdlambda_temp, spec_i, label=label_i,linestyle='--',alpha=0.5)
                    else:
                      ax1.plot(MIRgdlambda_temp, spec_i, label=label_i, color=colour_list[i], linestyle='--',alpha=0.5)
                    count += 1

        ax1.legend(ncol=2)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xticklabels([])
        ax1.set_ylim(1e-5,1e2)
        ax1.set_ylabel('Flux')

        ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
        ax2.plot(MIRgdlambda,MIRgdflux/MIRcontinuum,color='black')
        ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
        ax2.set_ylabel('Data/Model')
        ax2.set_xlabel('Wavelength [micron]')
        gs.update(wspace=0.0, hspace=0.05)

        if 'argscontfit' in initdat:
            if 'outdir' in initdat['argscontfit']:
                plt.savefig(initdat['argscontfit']['outdir']+initdat['label']+'_decomposition')
