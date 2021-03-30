import numpy as np
import lmfit
from q3dfit.common import interptemp
from q3dfit.common import questfitfcn
from q3dfit.common import questfit_readcf
from lmfit.models import ExpressionModel
from matplotlib.pyplot import *
from q3dfit.common import interp_temp_quest
from q3dfit.common import writeout_quest

#filename = '../test/example_cf.cf'
directory = '../test/test_questfit/'
filename = '../test/test_questfit/4978688_0.ideos.cf'
loc_models = '../data/questfit_templates/'
config_file = questfit_readcf.readcf(filename)


models_dictionary = {}
template_dictionary = {}
extinction_absorption_dictionary = {}


for i in config_file.keys(): #populating the models dictionary and setting up lmfit models
    if 'blackbody' in i: #starting with the blackbodies
        model_parameters = config_file[i]
        name_model = i
        extinction_model = config_file[i][3]
        ice_model = config_file[i][9]
        print(name_model,extinction_model,ice_model)



        model_temp_BB,param_temp_BB = questfitfcn.set_up_fit_blackbody_model([float(model_parameters[1]),float(model_parameters[7])],[float(model_parameters[2]),float(model_parameters[8])],name_model[:])

        model_temp_extinction,param_temp_extinction = questfitfcn.set_up_fit_extinction([float(model_parameters[4])],[float(model_parameters[5])],'blackbody'+str(int(float(model_parameters[7])))+'_ext',extinction_model,model_parameters[6])
                
        model_temp = model_temp_BB*model_temp_extinction
        param_temp = param_temp_BB + param_temp_extinction

        models_dictionary[extinction_model] = config_file[extinction_model]

        if 'ice' in i: #checking if we need to add ice absorption
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
        name_model = i
        extinction_model = config_file[i][3]
        ice_model = config_file[i][9]
        print(name_model,extinction_model,ice_model)

        model_temp_powerlaw,param_temp_powerlaw = questfitfcn.set_up_fit_powerlaw_model([1,float(model_parameters[7])],[float(model_parameters[2]),float(model_parameters[8])],name_model[:])

        model_temp_extinction,param_temp_extinction = questfitfcn.set_up_fit_extinction([float(model_parameters[4])],[float(model_parameters[5])],'powerlaw'+str(int(float(model_parameters[7])))+'_ext',extinction_model,model_parameters[6])
        
        model_temp = model_temp_powerlaw*model_temp_extinction
        param_temp = param_temp_powerlaw + param_temp_extinction

        models_dictionary[extinction_model] = config_file[extinction_model]

        if 'ice' in i: #checking if we need to add ice absorption
            model_temp_ice,param_temp_ice = questfitfcn.set_up_absorption([float(model_parameters[10])],[float(model_parameters[11])],name_model+'_abs',model_parameters[9])
            model_temp = model_temp*model_temp_ice
            param_temp += param_temp_ice
            models_dictionary[model_parameters[9]] = config_file[model_parameters[9]]

        if 'model' not in vars():
            model,param = model_temp,param_temp

        else:
            model += model_temp
            param += param_temp

    n_temp = 0
    if 'template' in i: #template model
        model_parameters = config_file[i]
        name_model = i
        extinction_model = config_file[i][3]
        ice_model = config_file[i][9]
        print(name_model,extinction_model,ice_model)

        model_temp_template,param_temp_template = questfitfcn.set_up_fit_model_scale([float(model_parameters[1])],[float(model_parameters[2])],name_model.split('.')[0],config_file[i][0])#name_model.split('.')[0]template+'_'+str(n_temp)
        
        if 'si' in i[0]:
            template_dictionary[config_file[i][0].split('.')[0]] = 'silicatemodels/'+config_file[i][0]
        else:
            template_dictionary[config_file[i][0].split('.')[0]] = config_file[i][0]
    
        model_temp_extinction,param_temp_extinction = questfitfcn.set_up_fit_extinction([float(model_parameters[4])],[float(model_parameters[5])],i.split('.')[0]+'_ext',extinction_model,model_parameters[6])
        
        model_temp = model_temp_template*model_temp_extinction
        param_temp = param_temp_template + param_temp_extinction

        models_dictionary[extinction_model] = config_file[extinction_model]

        if 'ice' in i: #checking if we need to add ice absorption
            model_temp_ice,param_temp_ice = questfitfcn.set_up_absorption([float(model_parameters[10])],[float(model_parameters[11])],name_model.split('.')[0]+'_abs',model_parameters[9])
            model_temp = model_temp*model_temp_ice
            param_temp += param_temp_ice
            models_dictionary[model_parameters[9]] = config_file[model_parameters[9]]

        if 'model' not in vars():
            model,param = model_temp,param_temp

        else:
            model += model_temp
            param += param_temp



data_to_fit = np.load(directory+config_file['source'][0],allow_pickle=True)
wave = data_to_fit['WAVE'].astype('float')[37:350]#np.arange(6,30,0.01)
flux = data_to_fit['FLUX'].astype('float')[37:350]
weights = data_to_fit['eflux'].astype('float')[37:350]

for i in models_dictionary.keys():
    temp_model = np.load(loc_models+models_dictionary[i],allow_pickle=True)
    temp_wave = []
    temp_value = []
    
    temp_wave=temp_model['WAVE']
    temp_value=temp_model['FLUX']


    #temp_value_rebin = interptemp.interptemp(wave_temp,temp_wave,temp_value)
    temp_value_rebin = interp_temp_quest.interp_lis(wave, temp_wave, temp_value)
    models_dictionary[i] = temp_value_rebin#/temp_value_rebin.max()


for i in template_dictionary.keys():
    temp_model = np.load(loc_models+template_dictionary[i],allow_pickle=True)
    temp_wave = []
    temp_value = []

    temp_wave=temp_model['WAVE']
    temp_value=temp_model['FLUX']
    
    
    #temp_value_rebin = interptemp.interptemp(wave_temp,temp_wave,temp_value)
    temp_value_rebin = interp_temp_quest.interp_lis(wave, temp_wave, temp_value)
    models_dictionary[i] = temp_value_rebin/temp_value_rebin.max()



models_dictionary['wave'] = wave/(1+0.04147)

eval = model.eval(param,**models_dictionary)

#model_temp_extinction.eval(params=param_temp_extinction,DRAINE03 = models_dictionary['DRAINE03'])

comp=dict(model.eval_components(params=param,**models_dictionary))

plot(wave,np.log10(eval),label='total_model')
for i in comp.keys():
    plot(wave,np.log10(comp[i]),label=i,linestyle='--',alpha=0.5)


#plot(wave_temp,np.log10(comp['blackbody_cold_DRAINE03_H2ice_']),label='blackbody_cold_DRAINE03_H2ice_')
#plot(wave_temp,np.log10(comp['blackbody_hot_DRAINE03_H2ice_']),label='blackbody_hot_DRAINE03_H2ice_')
legend()
xlabel('wavelength [micron]')


result = model.fit(flux,param,**models_dictionary,max_nfev=int(1e5),method='least_squares',nan_policy='omit')#method='least_squares'nan_policy='omit'

best_fit = result.eval(**models_dictionary)
comp_best_fit = result.eval_components()

close("all")
plot(wave,flux,color='black')
plot(wave,best_fit)

for i in np.arange(0,len(comp.keys()),3):
    plot(wave,comp_best_fit[list(comp.keys())[i]]*comp_best_fit[list(comp.keys())[i+1]]*comp_best_fit[list(comp.keys())[i+2]],label=list(comp.keys())[i],linestyle='--',alpha=0.5)
legend()
#plot(wave,comp_best_fit['template_smith_nftemp3'])
#plot(wave,comp_best_fit['template_smith_nftemp4'])


xscale('log')
yscale('log')
#template    smith_nftemp3.npy    1.0    1.    Chiar06    0.0    1.0    S    0.0    0.0 ice_hc    0.0    1.0
#template    smith_nftemp4.npy    1.0    1.    Chiar06    0.0    1.0    S    0.0    0.0 ice_hc    0.0    1.0

show()


# -- Save output --

writeout_quest.save_spectral_comp(wave, flux, best_fit, comp_best_fit, filename)
writeout_quest.save_params(result, filename)



