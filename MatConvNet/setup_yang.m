code_dirs={'linear_classifier', 'layers', 'get_batch', ...
    'get_activations', 'prepare_dataset'};
for i=1:numel(code_dirs)
    addpath(genpath(code_dirs{i}));
end

run(fullfile('vlfeat','toolbox','vl_setup'));
run(fullfile('matconvnet','matlab','vl_setupnn'));

addpath(fullfile('matconvnet','examples'));

% gtsvm on my machine
addpath('/home/yang/local/my_software/gtsvm/mex')
