import numpy as np
import lmfit
from q3dfit.common import interptemp
from q3dfit.common import questfitfcn
from q3dfit.common import questfit_readcf
from lmfit.models import ExpressionModel
from matplotlib.pyplot import *
filename = '../test/example_cf.cf'
loc_models = '../data/questfit_templates/'
config_file = questfit_readcf.readcf(filename)


models_dictionary = {}
extinction_absorption_dictionary = {}


for i in config_file.keys(): #populating the models dictionary and setting up lmfit models
    if 'blackbody' in i: #starting with the blackbodies
        model_parameters = config_file[i]
        name_model = i
        extinction_model = config_file[i][3]
        ice_model = config_file[i][9]
        print(name_model,extinction_model,ice_model)

        model_temp_BB,param_temp_BB = questfitfcn.set_up_fit_blackbody_model([1,float(model_parameters[7])],[float(model_parameters[2]),float(model_parameters[8])],name_model[:])

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


wave_temp = np.arange(6,30,0.01)

for i in models_dictionary.keys():
    temp_model = np.load(loc_models+models_dictionary[i],allow_pickle=True)
    temp_wave = []
    temp_value = []
    for j in temp_model:
        temp_wave.append(j[0])
        temp_value.append(j[1])


    temp_value_rebin = interptemp.interptemp(wave_temp,np.array(temp_wave),np.array(temp_value))

    models_dictionary[i] = temp_value_rebin

models_dictionary['wave'] = wave_temp

eval = model.eval(param,**models_dictionary)

#model_temp_extinction.eval(params=param_temp_extinction,DRAINE03 = models_dictionary['DRAINE03'])

comp=dict(model.eval_components(params=param,**models_dictionary))

plot(wave_temp,np.log10(eval),label='total_model')
for i in comp.keys():
    plot(wave_temp,np.log10(comp[i]),label=i,linestyle='--',alpha=0.5)


#plot(wave_temp,np.log10(comp['blackbody_cold_DRAINE03_H2ice_']),label='blackbody_cold_DRAINE03_H2ice_')
#plot(wave_temp,np.log10(comp['blackbody_hot_DRAINE03_H2ice_']),label='blackbody_hot_DRAINE03_H2ice_')
legend()
xlabel('wavelength [micron]')

