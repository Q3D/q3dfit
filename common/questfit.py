import numpy as np
import lmfit
import copy
from scipy import constants
from q3dfit.common import interptemp
from q3dfit.common import questfitfcn
from q3dfit.common import questfit_readcf
from lmfit.models import ExpressionModel
from matplotlib import pyplot as plt
from q3dfit.common import interp_temp_quest
from q3dfit.common import writeout_quest
import q3dfit



def questfit(wlambda, flux, weights, singletemplatelambda, singletemplateflux, index, 
    z, quiet=True, config_file=None, global_ice_model='None', global_ext_model='None', \
    models_dictionary={}, template_dictionary={}, fitran=None, convert2Flambda=True):
    '''Function defined to fit the MIR continuum

    Parameters
    -----
    wlambda: array
        wavelength

    flux: array
        Flux values to be fit

    weight: array
        Weights of each individual pixel to be fit

    singletemplatelambda: array
        Disregarded if set to b'0'. 
        Otherwise, this is a wavelength array for any continuum template separate from the simple empirical BB, power-law etc. components.

    singletemplateflux: array
        Disregarded if set to b'0'. 
        Otherwise, this is a flux array for any continuum template separate from the simple empirical BB, power-law etc. components.

    index: array
        Pixels used in the fit

    z: float
        redshift



    returns
    -------
    continuum: array
        best fit continuum model

    ct_coeff: dictionary or lmfit best fit params structure
        best fit parameters

    '''
    

    models_dictionary = {}
    template_dictionary = {}

    if fitran:
        flux = flux[ np.logical_and(wlambda>=fitran[0]), np.logical_and(wlambda<=fitran[1]) ]
        wlambda = wlambda[ np.logical_and(wlambda>=fitran[0]), np.logical_and(wlambda<=fitran[1]) ]
    if global_ext_model != 'None':     global_extinction = True


    if singletemplatelambda!=b'0':
        print('Trying to pass a single separate template to questfit, which is not implemented ... Halting.')
        import sys; sys.exit()

    else:
        config_file = questfit_readcf.readcf(config_file)
        loc_models = q3dfit.__path__[0]+'/data/questfit_templates/'
        n_temp = 0
        
        for i in config_file.keys(): #populating the models dictionary and setting up lmfit models

            if 'blackbody' in i: #starting with the blackbodies
                model_parameters = config_file[i]
                name_model = 'blackbody'+str(int(float(model_parameters[7])))#i
                extinction_model = config_file[i][3]
                ice_model = config_file[i][9]

                model_temp_BB,param_temp_BB = questfitfcn.set_up_fit_blackbody_model([float(model_parameters[1]),float(model_parameters[7])],[float(model_parameters[2]),float(model_parameters[8])],name_model[:])
                
                if global_extinction == False:
                    model_temp_extinction,param_temp_extinction = questfitfcn.set_up_fit_extinction([float(model_parameters[4])],[float(model_parameters[5])],name_model+'_ext',extinction_model,model_parameters[6])
                    
                    model_temp = model_temp_BB*model_temp_extinction
                    param_temp = param_temp_BB + param_temp_extinction

                    models_dictionary[extinction_model] = config_file[extinction_model]
                
                else:
                    model_temp = model_temp_BB
                    param_temp = param_temp_BB
                
                if 'ice' in i and global_extinction == False: #checking if we need to add ice absorption
                    model_temp_ice,param_temp_ice = questfitfcn.set_up_absorption([float(model_parameters[10])],[float(model_parameters[11])],name_model+'_abs',model_parameters[9])
                    model_temp = model_temp*model_temp_ice
                    param_temp += param_temp_ice
                    models_dictionary[model_parameters[9]] = config_file[model_parameters[9]]
                if 'model' not in vars():
                    model,param = model_temp,param_temp

                else:
                    model += model_temp
                    param += param_temp


            if 'powerlaw' in i: #powerlaw model
                model_parameters = config_file[i]
                name_model = 'powerlaw'+str(int(float(model_parameters[7])))
                extinction_model = config_file[i][3]
                ice_model = config_file[i][9]

                model_temp_powerlaw,param_temp_powerlaw = questfitfcn.set_up_fit_powerlaw_model([1,float(model_parameters[7])],[float(model_parameters[2]),float(model_parameters[8])],name_model[:])

                model_temp_extinction,param_temp_extinction = questfitfcn.set_up_fit_extinction([float(model_parameters[4])],[float(model_parameters[5])],'powerlaw'+str(int(float(model_parameters[7])))+'_ext',extinction_model,model_parameters[6])
                
                model_temp = model_temp_powerlaw*model_temp_extinction
                param_temp = param_temp_powerlaw + param_temp_extinction

                models_dictionary[extinction_model] = config_file[extinction_model]

                if 'ice' in i and global_extinction == False: #checking if we need to add ice absorption
                    model_temp_ice,param_temp_ice = questfitfcn.set_up_absorption([float(model_parameters[10])],[float(model_parameters[11])],name_model+'_abs',model_parameters[9])
                    model_temp = model_temp*model_temp_ice
                    param_temp += param_temp_ice
                    models_dictionary[model_parameters[9]] = config_file[model_parameters[9]]

                if 'model' not in vars():
                    model,param = model_temp,param_temp

                else:
                    model += model_temp
                    param += param_temp


            if 'template' in i: #template model
                model_parameters = config_file[i]
                name_model = 'template_'+str(n_temp)#i
                extinction_model = config_file[i][3]
                ice_model = config_file[i][9]

                model_temp_template,param_temp_template = questfitfcn.set_up_fit_model_scale([float(model_parameters[1])],[float(model_parameters[2])],name_model,name_model)#name_model.split('.')[0]template+'_'+str(n_temp)
                
                if 'si' in i:
                    #config_file[i][0].split('.')[0]
                    template_dictionary[name_model] = 'silicatemodels/'+config_file[i][0]
                else:

                    template_dictionary[name_model] = config_file[i][0]
            
                if global_extinction == False:

                    model_temp_extinction,param_temp_extinction = questfitfcn.set_up_fit_extinction([float(model_parameters[4])],[float(model_parameters[5])],name_model+'_ext',extinction_model,model_parameters[6])
                
                    model_temp = model_temp_template*model_temp_extinction
                    param_temp = param_temp_template + param_temp_extinction

                    models_dictionary[extinction_model] = config_file[extinction_model]

                else:
                    model_temp = model_temp_template
                    param_temp = param_temp_template

                if 'ice' in i and global_extinction == False: #checking if we need to add ice absorption
                    model_temp_ice,param_temp_ice = questfitfcn.set_up_absorption([float(model_parameters[10])],[float(model_parameters[11])],name_model+'_abs',model_parameters[9])
                    model_temp = model_temp*model_temp_ice
                    param_temp += param_temp_ice
                    models_dictionary[model_parameters[9]] = config_file[model_parameters[9]]

                if 'model' not in vars():
                    model,param = model_temp,param_temp

                else:
                    model += model_temp
                    param += param_temp

                n_temp+=1
        if global_extinction == True: #Check to see if we are using global extinction, where the total model flux is extincted by the same ice and dust model.

            model_global_ext,param_global_ext = questfitfcn.set_up_fit_extinction([0],[1],'global_ext',global_ext_model,'S')
            model = model*model_global_ext
            param += param_global_ext
            models_dictionary[extinction_model] = config_file[global_ext_model]

            model_global_ice,param_global_ice = questfitfcn.set_up_absorption([0],[1],'global_ice',global_ice_model)

            model = model*model_global_ice
            param += param_global_ice
            models_dictionary[ice_model] = config_file[global_ice_model]


        for i in models_dictionary.keys(): #loop over models dictionary, load them in and resample.
            temp_model = np.load(loc_models+models_dictionary[i],allow_pickle=True)

            temp_wave = []
            temp_value = []
            
            temp_wave=temp_model['WAVE']
            temp_value=temp_model['FLUX']
            
            
            #temp_value_rebin = interptemp.interptemp(wave_temp,temp_wave,temp_value)
            #temp_value_rebin = interp_temp_quest.interp_lis(wave, temp_wave, temp_value)
            temp_value_rebin = interp_temp_quest.interp_lis(wlambda, temp_wave, temp_value)
            models_dictionary[i] = temp_value_rebin#/temp_value_rebin.max()


        for i in template_dictionary.keys(): #loop over template dictionary, load them in and resample.

            temp_model = np.load(loc_models+template_dictionary[i],allow_pickle=True)
            temp_wave = []
            temp_value = []
            
            temp_wave=temp_model['WAVE']
            temp_value=temp_model['FLUX']
            
            
            #temp_value_rebin = interptemp.interptemp(wave_temp,temp_wave,temp_value)
            temp_value_rebin = interp_temp_quest.interp_lis(wlambda, temp_wave, temp_value)
            models_dictionary[i] = temp_value_rebin/temp_value_rebin.max()

        models_dictionary['wave'] = wlambda/(1+z)
        
        if convert2Flambda:
            flux *= constants.c/(wlambda*1e-9)**2 *1e-23
            for el in models_dictionary.keys():
                if not (global_ext_model in el) and not (global_ice_model in el) and not ('ext' in el) and not ('ice' in el):
                    models_dictionary[el]*= constants.c/(wlambda*1e-9)**2 *1e-23

        flux_cut = flux[index]
        models_dictionary_cut = copy.deepcopy(models_dictionary)
        for el in models_dictionary.keys():
            models_dictionary_cut[el] = models_dictionary_cut[el][index]


        result = model.fit(flux_cut,param,**models_dictionary_cut,max_nfev=int(1e5),method='least_squares',nan_policy='omit')#method='least_squares'nan_policy='omit'

        best_fit = result.eval(**models_dictionary) # use models_dictionary rather than models_dictionary_cut to evaluate over all wavelengths within fitran (not just [index])
        comp_best_fit = result.eval_components(**models_dictionary)

        if convert2Flambda:
            flux /= (constants.c/(wlambda*1e-9)**2 *1e-23)
            best_fit /= (constants.c/(wlambda*1e-9)**2 *1e-23)
            for el in comp_best_fit.keys():
                if not (global_ext_model in el) and not (global_ice_model in el) and not ('ext' in el) and not ('ice' in el):
                    try:
                        comp_best_fit[el] /= (constants.c/(wlambda*1e-9)**2 *1e-23)
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()

        ct_coeff = {'MIRparams': result.params, 'comp_best_fit': comp_best_fit}

        #return best_fit,comp_best_fit,result
        return best_fit, ct_coeff, z


do_test = False
if do_test:
    directory = '../test/test_questfit/'
    loc_models = '../data/questfit_templates/'
    models_dictionary = {}
    template_dictionary = {}
    extinction_absorption_dictionary = {}

    global_extinction = True
    global_ice_model = 'ice_hc'
    global_ext_model = 'Chiar06'



    #For testing MRK 231
    # z=0.04147
    # filename = '../test/test_questfit/4978688_0.ideos.cf'
    # config_file = questfit_readcf.readcf(filename)
    # data_to_fit = np.load(directory+config_file['source'][0],allow_pickle=True)
    # wave = data_to_fit['WAVE'].astype('float')[37:350]#np.arange(6,30,0.01)
    # flux = data_to_fit['FLUX'].astype('float')[37:350]
    # weights = data_to_fit['eflux'].astype('float')[37:350]

    #For testing IRAS21219
    filename = '../test/test_questfit/IRAS21219m1757_dlw_qst.cf'
    config_file = questfit_readcf.readcf(filename)
    z=0.112
    global_ext_model = 'CHIAR06'
    data_to_fit = np.load(directory+config_file['source'][0],allow_pickle=True)[0]
    wave = data_to_fit['WAVE'].astype('float')
    wave_min = np.where(wave>=float(config_file['source'][1]))[0][0]
    wave_max = np.where(wave>=float(config_file['source'][2]))[0][0]

    wave = data_to_fit['WAVE'].astype('float')[wave_min:wave_max]
    flux = data_to_fit['FLUX'].astype('float')[wave_min:wave_max]
    weights = data_to_fit['stdev'].astype('float')[wave_min:wave_max]
    


    singletemplatelambda = b'0'
    singletemplateflux = b'0'
    index = None
    best_fit, MIRct_coeff, MIRz = questfit(wave,flux,weights,singletemplatelambda, singletemplateflux, \
        index,z, quiet=True, config_file=filename, global_ice_model=global_ice_model, global_ext_model=global_ext_model, \
        models_dictionary={}, template_dictionary={}, fitran=None)

    if len(best_fit)==1:    best_fit=best_fit[0]

    comp_best_fit = MIRct_coeff['comp_best_fit']



    #eval = model.eval(param,**models_dictionary)

    #model_temp_extinction.eval(params=param_temp_extinction,DRAINE03 = models_dictionary['DRAINE03'])

    #comp=dict(model.eval_components(params=param,**models_dictionary))
    #
    #fig = plt.figure(figsize=(12, 7))
    #gs = fig.add_gridspec(2,1)
    #ax1 = fig.add_subplot(gs[0, :])
    #ax1.plot(wave,np.log10(eval),label='total_model')
    #for i in comp.keys():
    #    ax1.plot(wave,np.log10(comp[i]),label=i,linestyle='--',alpha=0.5)


    #plt.plot(wave_temp,np.log10(comp['blackbody_cold_DRAINE03_H2ice_']),label='blackbody_cold_DRAINE03_H2ice_')
    #plt.plot(wave_temp,np.log10(comp['blackbody_hot_DRAINE03_H2ice_']),label='blackbody_hot_DRAINE03_H2ice_')
    #ax1.legend()
    #ax1.set_xlabel('wavelength [micron]')




    #plt.close("all")



    fig = plt.figure(figsize=(6, 7))
    gs = fig.add_gridspec(4,1)
    ax1 = fig.add_subplot(gs[:3, :])

    ax1.plot(wave,flux,color='black')
    ax1.plot(wave,best_fit)

    if global_extinction == True:
       for i in np.arange(0,len(comp_best_fit.keys())-2,1):
           try:
              ax1.plot(wave,comp_best_fit[list(comp_best_fit.keys())[i]]*comp_best_fit[list(comp_best_fit.keys())[-2]]*comp_best_fit[list(comp_best_fit.keys())[-1]],label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)
           except:
              ax1.plot(wave,comp_best_fit[list(comp_best_fit.keys())[i]][0]*comp_best_fit[list(comp_best_fit.keys())[-2]][0]*comp_best_fit[list(comp_best_fit.keys())[-1]][0],label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)

    if global_extinction == False:
       for i in np.arange(0,len(comp_best_fit.keys()),3):
           ax1.plot(wave,comp_best_fit[list(comp_best_fit.keys())[i]]*comp_best_fit[list(comp_best_fit.keys())[i+1]]*comp_best_fit[list(comp_best_fit.keys())[i+2]],label=list(comp_best_fit.keys())[i],linestyle='--',alpha=0.5)

    counter = 0

    # while counter < len(comp_best_fit.keys()):
    #     if list(comp_best_fit.keys())[counter]+'_abs' in list(comp_best_fit.keys()):
    #         ax1.plot(wave,comp_best_fit[list(comp_best_fit.keys())[counter]]*comp_best_fit[list(comp_best_fit.keys())[counter+1]]*comp_best_fit[list(comp_best_fit.keys())[counter+2]],label=list(comp_best_fit.keys())[counter],linestyle='--',alpha=0.5)
    #         counter+=3

    #     else:
    #         ax1.plot(wave,comp_best_fit[list(comp_best_fit.keys())[counter]]*comp_best_fit[list(comp_best_fit.keys())[counter+1]],label=list(comp_best_fit.keys())[counter],linestyle='--',alpha=0.5)
    #         counter+=2


    ax1.legend(ncol=2)

    #plt.plot(wave,comp_best_fit['template_smith_nftemp3'])
    #plt.plot(wave,comp_best_fit['template_smith_nftemp4'])


    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticklabels([])
    ax1.set_ylim(1e-4,1e2)
    #template    smith_nftemp3.npy    1.0    1.    Chiar06    0.0    1.0    S    0.0    0.0 ice_hc    0.0    1.0
    #template    smith_nftemp4.npy    1.0    1.    Chiar06    0.0    1.0    S    0.0    0.0 ice_hc    0.0    1.0

    ax2 = fig.add_subplot(gs[-1, :], sharex=ax1)
    ax2.plot(wave,flux/best_fit,color='black')
    ax2.axhline(1, color='grey', linestyle='--', alpha=0.7, zorder=0)
    ax2.set_ylabel('Data/Model')
    ax2.set_xlabel('Wavelength [micron]')
    gs.update(wspace=0.0, hspace=0.05)

    plt.show()

    # fig = gcf()
    # gs = fig.add_gridspec(2,1)
    # ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)

    # -- Save output --

    #writeout_quest.save_spectral_comp(wave, flux, best_fit, comp_best_fit, filename)
    #writeout_quest.save_params(result, filename)


    # Reset
    models_dictionary = {}
    template_dictionary = {}
    extinction_absorption_dictionary = {}

