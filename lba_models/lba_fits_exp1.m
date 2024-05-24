clear all
close all
lba_folder = "lba_models/";

iti = [0,500,2000]; %[2:6];
Ns_values = iti;
Nc = length(iti);
Na = 4;
sv = 0.1; % Parameter identifiability

cmap = brewermap(7, 'Set1');
cmap = cmap([1,3,2,4,5,6],:);

experiment = "exp1";
%% Parsing human data for LBA fits
data_human_exp1 = load('iti_bandit_data_exp1.mat','datas');
data_human_exp1 = data_human_exp1.datas.test;
datas = parsedata_forlba(experiment, data_human_exp1, iti);
%save(lba_folder+'data/lba_data_exp1_struct', 'datas');

%% Human LBA fits
is_paramrecov = false;
% 0) Load human data
datas = load(lba_folder+'data/lba_data_exp1_struct').datas;
% 1) Specify LBA models. 
% Note: "model.sv" is actually v_incorrect now!
model(1).v = 1; model(1).A = 1; model(1).b = 1; model(1).t0 = 1; model(1).sv = 1;
model(2).v = 1; model(2).A = 1; model(2).b = Nc; model(2).t0 = 1; model(2).sv = 1;
model(3).v = Nc; model(3).A = Nc; model(3).b = Nc; model(3).t0 = Nc; model(3).sv = Nc;
names{1} = ['v\_correct \t A \t B-A \t v\_incorrect \t t0'];
names{2} = ['v\_correct \t A \t B_1-A \t B_2-A \t B_3-A \t v\_incorrect \t t0'];
names{3} = ['v\_correct_1 \t v\_correct_2 \t v\_correct_3 \t A_1 \t A_2 \t A_3 \t B_1-A_1 \t B_2-A_2 \t B_3-A_3 \t v\_incorrect_1 \t v\_incorrect_2 \t v\_incorrect_3 \t t0_1 \t t0_2 \t t0_3'];

model12_fit_filename = "lba_exp1_withoutmodel3";
model3_fit_filename = "lba_exp1_model3";
models_all_fit_filename = "lba_exp1_full";

% 2) Fit LBAs to human data.
n_inits=60; %60

[Params_best, Params, LLs, pArray] = fit_lba_models(experiment, datas, model, n_inits, [1,2], iti, Na, is_paramrecov);
save(lba_folder+"fits/"+model12_fit_filename, 'model', 'pArray','names','Params','LLs','Params_best')
[Params_best, Params, LLs, pArray] = fit_lba_models(experiment, datas, model, n_inits, 3, iti, Na, is_paramrecov);
save(lba_folder+"fits/"+model3_fit_filename, 'model', 'pArray','names','Params','LLs','Params_best')

% 3) Merge LBA fit files
[model, pArray,names,Params,LLs, ~, ~, Params_best,LLs_model3_sep] = merge_lbafits(model12_fit_filename, model3_fit_filename, lba_folder, is_paramrecov);
save(lba_folder+"fits/"+models_all_fit_filename, 'model', 'pArray','names','Params','LLs','Params_best','LLs_model3_sep')






%% Helper functions: convert datafile to the format used by LBA code
function [datas] = parsedata_forlba(experiment, data_orig, tasks)
% This code takes in the data file and converts it to the format used by
% LBA code. 
    if(ismember("exp3", experiment))
        n_subj = length(data_orig.("set_size_2"));
    else
        n_subj = length(data_orig)
    end
    datas = cell(n_subj,1);
    for subj=1:n_subj
        "Subject "+num2str(subj)+": parse data"
        cond=[];
        correct=[];
        stim=[];
        response=[];
        rt=[];
        for c=1:length(tasks) % ITI for Experiment 1, set_size for Experiment 3. 
            task_val = tasks(c);
            if(ismember("exp3",experiment))
                data_task_subj = data_orig.("set_size_"+task_val)(subj);
            else
                data_task_subj = data_orig(subj);
                idx = find(data_task_subj.cond == task_val);
                data_task_subj.rt = data_task_subj.rt(idx);
                data_task_subj.s = data_task_subj.s(idx);
                data_task_subj.a = data_task_subj.a(idx);
                data_task_subj.acc = data_task_subj.acc(idx);
            end
    
            num_trials_task_subj = length(data_task_subj.rt);
            rt=[rt; data_task_subj.rt];
            stim=[stim; data_task_subj.s];
            response=[response; data_task_subj.a];
            cond=[cond; c.*ones(num_trials_task_subj,1)];
            correct=[correct; data_task_subj.acc];
        end
        data.cond=cond;
        data.stim=stim;
        data.response=response;
        data.rt=rt;
        data.correct=correct;
        datas{subj} = data;
    end
end


function [Params_best, Params, LLs, pArray] = fit_lba_models(experiment, Datas, model, n_inits, model_idxs, task_vals, Na, is_paramrecov)
    % Given an LBA-compatible dataset and model specifications (in models), fit the models with indexing model_idxs (either [1,2] or [3]).  
    if(is_paramrecov)
        n_subj = length(Datas{1});
        models = length(Datas);
    else
        n_subj = length(Datas);
        models = 3;
    end

    Nc = length(task_vals);
    % Note: model.sv is actually v_incorrect now!
    pArray = cell(models,n_inits);
    for init=1:n_inits
        pArray{1,init} = rand(1,5).*[2,500,200,2,0.1]; %[0.8 300 150 0.4 1];
        if(ismember("exp3", experiment)) % Experiment 3, LBA 2 varies the mean drift rates v_correct, v_incorrect.
            pArray{2,init} = rand(1,2*Nc+3).*[repmat(2,1,Nc),500,200,2,repmat(0.1,1,Nc)]; %[0.8 0.8 0.8 0.8 0.8 300 150 0.4 0.4 0.4 0.4 0.4 1];
        else % Experiment 1, LBA 2 varies the bound b.
            pArray{2,init} = rand(1,Nc+4).*[2,500,repmat(200,1,Nc),2,0.1]; %[0.8 0.8 0.8 0.8 0.8 300 150 0.4 0.4 0.4 0.4 0.4 1];
        end
        pArray{3,init} = rand(1,5*Nc).*[repmat(2,1,Nc),repmat(500,1,Nc),repmat(200,1,Nc),repmat(2,1,Nc),repmat(0.1,1,Nc)]; %[repmat(0.8,1,5), repmat(300,1,5) repmat(150,1,5) repmat(0.4,1,5) repmat(1,1,5)];
    end
    
    Params = cell(n_subj,1);
    LLs = cell(n_subj,1);
    for subj=1:n_subj
        "Subject "+num2str(subj)+":"
        % 1) Clean up data (I cannot clean up it for our purpose--most trials have RT < 200ms. 
        % data = LBA_clean(data); %
        for m = model_idxs
            if(is_paramrecov)
                datas = Datas{m};
            else
                datas = Datas;
            end
            parfor init=1:n_inits   
                [subj, m, init]
                params_Temp = zeros(1, length(pArray{3,1}));
                if(m~=3)
                    [params{m,init}, LL(m,init)] = LBA_mle(datas{subj}, model(m), pArray{m,init}, Na);
                else % Fit independent LBA for each set size condition
                    ll=[];
                    for c = 1:length(task_vals)
                        p_array = pArray{m,init}(c:length(task_vals):end); 
                        relevant_trialidx_cond = find(datas{subj}.cond==c);
    
                        data = datas{subj}; 
                        data.cond = ones(length(relevant_trialidx_cond),1);
                        data.correct = datas{subj}.correct(relevant_trialidx_cond);
                        data.stim = datas{subj}.stim(relevant_trialidx_cond);
                        data.response = datas{subj}.response(relevant_trialidx_cond);
                        data.rt = datas{subj}.rt(relevant_trialidx_cond);
    
                        [params_temp, ll_temp] = LBA_mle(data, model(1), p_array, Na);
                        params_Temp(c:length(task_vals):end) = params_temp;
                        ll(c) = ll_temp;
                    end
                    params{m,init} = params_Temp; LL{m,init}=ll;
                end
            end
        end
        Params{subj}=params;
        LLs{subj}=LL;
    end

    % Find best fit across all inits for each subject and model. 
    Params_best = cell(n_subj,1);
    if(~ismember(3, model_idxs)) % Use when model_idxs = [1,2]
        for subj=1:n_subj
            [~, loglike_max_init] = max(LLs{subj},[],2);
            params_best = cell(3,1);
            for m=1:2   
                params_best{m} = Params{subj}{m,loglike_max_init(m)};
            end
            Params_best{subj} = params_best;
        end
    else % Use when model_idxs = 3
        for subj=1:n_subj
            params_best_subj_all_models = cell(3,1);
            LLs_subj = LLs{subj};
            params_subj = Params{subj};
            lls_subj_model3 = zeros(n_inits,length(task_vals));
            params_best_subj = zeros(1,Nc*5);
            for init=1:n_inits
                lls_subj_model3(init,:) = LLs_subj{3,init}; % Rows are different inits, Cols are different set size conds.
            end
            [~, loglike_max_init] = max(lls_subj_model3,[],1);
            for c=1:length(task_vals)
                val = params_subj{3,loglike_max_init(c)};
                params_best_subj(c:length(task_vals):end) = val(c:length(task_vals):end);
            end
            params_best_subj_all_models{3} = params_best_subj;
            Params_best{subj} = params_best_subj_all_models;
        end
    end
end


function [model, pArray,names,Params,LLs, Datas, True_Params, Params_best,LLs_model3_sep] = merge_lbafits(model12_fit_filename, model3_fit_filename, lba_folder, is_paramrecov)
    % For the saved LBA fits for Model 1, 2 vs. 3, merge their saved files.
    load(lba_folder+"fits/"+model3_fit_filename)
    LLs_sep = LLs;
    Params_sep = Params;
    pArray_sep = pArray;
    Params_best_sep = Params_best;
    
    load(lba_folder+"fits/"+model12_fit_filename)
    n_subj=length(LLs);
    n_inits=size(Params{1});
    n_inits = n_inits(2);
    for init=1:n_inits
        pArray{3,init} = pArray_sep{3,init};
    end
    for subj=1:n_subj
        for init=1:n_inits
            %LLs{subj}(3,init) = LLs_sep{subj}{3,init};
            Params{subj}{3,init} = Params_sep{subj}{3,init};
        end
    end
    
    Params_best = cell(n_subj,1);
    for subj=1:n_subj
        [~, loglike_max_init] = max(LLs{subj},[],2);
        params_best = cell(3,1);
        for m=1:2
            params_best{m} = Params{subj}{m,loglike_max_init(m)};
        end
        params_best{3} = Params_best_sep{subj}{3};
        Params_best{subj} = params_best;
    end
    LLs_model3_sep = LLs_sep;

    if(~is_paramrecov)
        True_Params = NaN;
        Datas = NaN;
    end
end
