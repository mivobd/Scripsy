handle=get_param('CSTR_Simulink','handle');
print(handle,'-dsvg','MyModel');
winopen MyModel.svg;

handle=get_param('CSTR_Simulink/Subsystem6','handle');
print(handle,'-dsvg','MyModel');
winopen MyModel.svg;