clear all
% base_folder = 'C:\Users\liu_s\policycompression_timememorycosts';
% cd(base_folder)
% addpath(genpath(base_folder))
lba_folder = "lba_models/";

%% Parsing human data for LBA fits

% data_human_exp1 = load('iti_bandit_data_exp1.mat','datas');
% data_human_exp1 = data_human_exp1.datas.test;
% datas = parsedata_forlba(experiment, data_human_exp1, iti);
% %save(lba_folder+'data/lba_data_exp1_struct', 'datas');

%% LBA fit specifications
is_paramrecov = false;

% 1) Specify LBA models. 
Nc = 3;
% Note: "model.sv" is actually v_incorrect now!

model(1).v = 1; model(1).A = 1; model(1).b = 1; model(1).t0 = 1; model(1).sv = 1;
model(2).v = 1; model(2).A = 1; model(2).b = Nc; model(2).t0 = 1; model(2).sv = 1;
model(3).v = Nc; model(3).A = Nc; model(3).b = Nc; model(3).t0 = Nc; model(3).sv = Nc;
model(4).v = 1; model(4).A = 1; model(4).b = Nc; model(4).t0 = 1; model(4).sv = 1; model(4).perserv = []; model(4).perservA = []; model(4).v_lrate = 3; model(4).p_lrate = 1;

names{1} = ['v\_correct \t A \t B-A \t v\_incorrect \t t0'];
names{2} = ['v\_correct \t A \t B_1-A \t B_2-A \t B_3-A \t v\_incorrect \t t0'];
names{3} = ['v\_correct_1 \t v\_correct_2 \t v\_correct_3 \t A_1 \t A_2 \t A_3 \t B_1-A_1 \t B_2-A_2 \t B_3-A_3 \t v\_incorrect_1 \t v\_incorrect_2 \t v\_incorrect_3 \t t0_1 \t t0_2 \t t0_3'];
names{4} = ['v\_correct \t A \t B_1-A \t B_2-A \t B_3-A \t v\_incorrect \t t0 \t v_lrate_1 \t v_lrate_2 \t v_lrate_3 \t p_lrate'];

% Number of random initializations per subject.
n_inits = 60;

%% 2) Fit LBA models
% Iterate over experiments
for exp_idx = 1:3
    experiment = "exp"+exp_idx;
    % 0) Load human data
    datas = load(lba_folder+'data/lba_data_'+experiment+'_struct').datas;
    if(exp_idx<3)
        iti = [0,500,2000]; 
        Ns_values = iti;
        Nc = length(iti);
        Na = 4;
        sv = 0.1; % Parameter identifiability
        fit_lba_input = iti;
    else
        set_sizes = [2,4,6]; 
        Ns_values = set_sizes;
        Nc = length(set_sizes);
        Na = max(set_sizes);
        sv = 0.1; % Parameter identifiability   
        fit_lba_input = set_sizes;
    end
    models_allfit_filename = "lba_"+experiment+"_full_test";   

    % Fit all models for this experiment
    for m=1:4
        model_fit_filename = "lba_"+experiment+"_model"+m;   
        [Params_best, Params, LLs, pArray] = fit_lba_models(experiment, datas, model, n_inits, [m], fit_lba_input, Na, is_paramrecov);
        save(lba_folder+"fits/"+model_fit_filename, 'model', 'pArray','names','Params','LLs','Params_best')
    end

    % Merge all LBA fits saved files.
    [~, pArray,~,Params,LLs, ~, ~, Params_best,LLs_model3_sep] = merge_lba_fits("exp"+exp_idx, lba_folder, is_paramrecov);
    save(lba_folder+"fits/"+models_allfit_filename, 'model', 'pArray','names','Params','LLs','Params_best','LLs_model3_sep')
end

