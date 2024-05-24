clear all
close all
lba_folder = "lba_models/";


set_sizes = [2,4,6]; %[2:6];
Ns_values = set_sizes;
Nc = length(set_sizes);
Na = max(set_sizes);
sv = 0.1; % Parameter identifiability

cmap = brewermap(7, 'Set1');
cmap = cmap([1,3,2,4,5,6],:);

data_human = load(datapath+'iti_bandit_data_exp3.mat').datas.test;
clear data_human

experiment = "exp3";
 
%% Parsing human data for LBA fits
data_human_exp3 = load('iti_bandit_data_exp3.mat','datas');
data_human_exp3 = data_human_exp3.datas.test;
datas = parsedata_forlba(data_human_exp3, set_sizes);
save(lba_folder+'data/lba_data_exp3_struct', 'datas');

%% Human LBA fits
is_paramrecov = false;
% 0) Load human data
datas = load(lba_folder+'data/lba_data_exp3_struct').datas;
% 1) Specify LBA models. 
% Note: "model.sv" is actually v_incorrect now!
model(1).v = 1; model(1).A = 1; model(1).b = 1; model(1).t0 = 1; model(1).sv = 1;
model(2).v = Nc; model(2).A = 1; model(2).b = 1; model(2).t0 = 1; model(2).sv = Nc;
model(3).v = Nc; model(3).A = Nc; model(3).b = Nc; model(3).t0 = Nc; model(3).sv = Nc;
names{1} = ['v\_correct \t A \t B-A \t v\_incorrect \t t0'];
names{2} = ['v\_correct_1 \t v\_correct_2 \t v\_correct_3 \t A \t B-A \t v\_incorrect_1 \t v\_incorrect_2 \t v\_incorrect_3 \t t0'];
names{3} = ['v\_correct_1 \t v\_correct_2 \t v\_correct_3 \t A_1 \t A_2 \t A_3 \t B_1-A_1 \t B_2-A_2 \t B_3-A_3 \t v\_incorrect_1 \t v\_incorrect_2 \t v\_incorrect_3 \t t0_1 \t t0_2 \t t0_3'];

model12_fit_filename = "lba_exp3_withoutmodel3";
model3_fit_filename = "lba_exp3_model3";
models_all_fit_filename = "lba_exp3_full";

% 2) Fit LBAs to human data.
n_inits=60; %60

[Params_best, Params, LLs, pArray] = fit_lba_models(datas, model, n_inits, [1,2], set_sizes, Na, is_paramrecov);
save(lba_folder+"fits/"+model12_fit_filename, 'model', 'pArray','names','Params','LLs','Params_best')
[Params_best, Params, LLs, pArray] = fit_lba_models(datas, model, n_inits, 3, set_sizes, Na, is_paramrecov);
save(lba_folder+"fits/"+model3_fit_filename, 'model', 'pArray','names','Params','LLs','Params_best')

% 3) Merge LBA fit files
[model, pArray,names,Params,LLs, ~, ~, Params_best,LLs_model3_sep] = merge_lbafits(model12_fit_filename, model3_fit_filename, lba_folder, is_paramrecov);
save(lba_folder+"fits/"+models_all_fit_filename, 'model', 'pArray','names','Params','LLs','Params_best','LLs_model3_sep')


%% Helpful function: convert datafile to the format used by LBA code
function [datas] = parsedata_forlba(data_orig, set_sizes)
% This code takes in the data file and converts it to the format used by
% LBA code. 
    n_subj = length(data_orig.("set_size_2"));
    datas = cell(n_subj,1);
    for subj=1:n_subj
        "Subject "+num2str(subj)+": parse data"
        cond=[];
        correct=[];
        stim=[];
        response=[];
        rt=[];
        for set_size_idx=1:length(set_sizes)
            set_size = set_sizes(set_size_idx);
            data_setsize_subj = data_orig.("set_size_"+set_size)(subj);
    
            num_trials_setsize_subj = length(data_setsize_subj.rt);
            rt=[rt; data_setsize_subj.rt];
            stim=[stim; data_setsize_subj.s];
            response=[response; data_setsize_subj.a];
            cond=[cond; set_size_idx.*ones(num_trials_setsize_subj,1)];
            correct=[correct; data_setsize_subj.acc];
        end
        data.cond=cond;
        data.stim=stim;
        data.response=response;
        data.rt=rt;
        data.correct=correct;
        datas{subj} = data;
    end
end

function [Params_best, Params, LLs, pArray] = fit_lba_models(Datas, model, n_inits, model_idxs, set_sizes, Na, is_paramrecov)
    % Given an LBA-compatible dataset and model specifications (in models), fit the models with indexing model_idxs (either [1,2] or [3]).  
    if(is_paramrecov)
        n_subj = length(Datas{1});
        models = length(Datas);
    else
        n_subj = length(Datas);
        models = 3;
    end

    Nc = length(set_sizes);
    % Note: model.sv is actually v_incorrect now!
    pArray = cell(models,n_inits);
    for init=1:n_inits
        pArray{1,init} = rand(1,5).*[2,500,200,2,0.1]; %[0.8 300 150 0.4 1];
        pArray{2,init} = rand(1,2*Nc+3).*[repmat(2,1,Nc),500,200,2,repmat(0.1,1,Nc)]; %[0.8 0.8 0.8 0.8 0.8 300 150 0.4 0.4 0.4 0.4 0.4 1];
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
                    for set_size_idx = 1:length(set_sizes)
                        p_array = pArray{m,init}(set_size_idx:length(set_sizes):end); 
                        relevant_trialidx_cond = find(datas{subj}.cond==set_size_idx);
    
                        data = datas{subj}; 
                        data.cond = ones(length(relevant_trialidx_cond),1);
                        data.correct = datas{subj}.correct(relevant_trialidx_cond);
                        data.stim = datas{subj}.stim(relevant_trialidx_cond);
                        data.response = datas{subj}.response(relevant_trialidx_cond);
                        data.rt = datas{subj}.rt(relevant_trialidx_cond);
    
                        [params_temp, ll_temp] = LBA_mle(data, model(1), p_array, Na);
                        params_Temp(set_size_idx:length(set_sizes):end) = params_temp;
                        ll(set_size_idx) = ll_temp;
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
            lls_subj_model3 = zeros(n_inits,length(set_sizes));
            params_best_subj = zeros(1,Nc*5);
            for init=1:n_inits
                lls_subj_model3(init,:) = LLs_subj{3,init}; % Rows are different inits, Cols are different set size conds.
            end
            [~, loglike_max_init] = max(lls_subj_model3,[],1);
            for set_size_idx=1:length(set_sizes)
                val = params_subj{3,loglike_max_init(set_size_idx)};
                params_best_subj(set_size_idx:length(set_sizes):end) = val(set_size_idx:length(set_sizes):end);
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

%% Helper functions: LME fits.
function [LMEs] = LME_regression(Complexity, Response_time, Ns_values, model_names, cmap)
    [models,n_subj,~]=size(Complexity);
    LMEs = cell(models, 1);
    complexity_grid = 0:0.05:log2(max(Ns_values));
    figure;
    tiledlayout(models,4, 'Padding', 'none', 'TileSpacing', 'compact');
    for m=1:models
        complexity = squeeze(Complexity(m,:,:));
        response_time = squeeze(Response_time(m,:,:))./1000;
        complexity_flat = complexity(:); 
        response_time_flat = response_time(:);
        [complexity_sorted, complexity_sorted_idx] = sort(complexity_flat);
        response_time_sorted = response_time_flat(complexity_sorted_idx);
        % figure
        % plot(complexity_sorted, response_time_sorted,"k.", 'MarkerSize',10)
        % hold on
        % mov_window_avg_RT = smoothdata(response_time_sorted,"gaussian",100);
        % plot(complexity_sorted, mov_window_avg_RT, "r-", 'LineWidth',1)
        
        
        subject_id = repmat(1:n_subj, 1, length(Ns_values))';
        tbl = table(subject_id,complexity_flat,response_time_flat,'VariableNames',{'Subject','PolicyComplexity','RT'});
        lme = fitlme(tbl,'RT ~ PolicyComplexity + (1|Subject) + (PolicyComplexity-1|Subject)');
        LMEs{m} = lme;
        % STD of random effects
        [~,~,stats] = randomEffects(lme,'Alpha',0.01);
        q = dataset2cell(stats(1:n_subj,4));
        w = dataset2cell(stats((n_subj+1):end,4));
        random_effects_intercept_std = std(cell2mat(q(2:end)));
        random_effects_complexity_std = std(cell2mat(w(2:end)));
        
        
        RT_lme = predict(lme);
        nexttile;
        hold on
        plot(response_time_flat, RT_lme, "k.", "MarkerSize",10)
        max_RT = max(response_time_flat)*1.05;
        plot([0,max_RT],[0,max_RT],"r--", "LineWidth",1)
        xlim([0,min(log2(6),max_RT)])
        ylim([0,min(log2(6),max_RT)])
        if(m==(models))
            xlabel("RT, human or LBA generated (s)")
        end
        ylabel({model_names(m)+" LME fits", "RT_{pred} (s)"})
    
        nexttile;
        hold on;
        for subj=1:n_subj
            tbl_new = table(repmat(subj,1,length(complexity_grid))',complexity_grid','VariableNames',{'Subject','PolicyComplexity'});
            RT_lme_theoretical = predict(lme, tbl_new);
            plot(complexity_grid, RT_lme_theoretical, "-", "LineWidth",1, 'Color',[0,0,0,0.1])
        end
        if(m==(models))
            xlabel("Policy complexity (bits)")
        end
        ylabel("RT_{pred} (s)")
        ylim([0,4])
        xlim([0, max(complexity_grid)])
    
        nexttile;
        histogram(RT_lme-response_time_flat, 30, "Normalization","probability")
        if(m==(models))
            xlabel("RT_{pred} - RT (s)")
        end
        ylabel("Relative Frequency")
        nexttile;
        histogram((RT_lme-response_time_flat)./response_time_flat, 30, "Normalization","probability")
        if(m==(models))
            xlabel("(RT_{pred} - RT) / RT")
        end
        ylabel("Relative Frequency")
        sgtitle({"LME fits on human/LBA generated data", "RT ~ PolicyComplexity + (1|Subject) + (PolicyComplexity-1|Subject)"})
        
        
        % figure;
        % plot(response_time_flat,(RT_lme-response_time_flat), "k.")
        % ylim([-1,1])
        % hold on
        % plot([0,min(2,max(response_time_flat))],[0,0],"r-")
        % ylabel("(RT_{pred}-RT_{true}) (s)")
        % xlabel("RT_{true} (s)")
    end
    

    % Subject-specific RT to policy complexity relationship
    for m=1:models
        figure
        tiledlayout(4,n_subj/4, 'Padding', 'none', 'TileSpacing', 'compact');

        complexity = squeeze(Complexity(m,:,:));
        response_time = squeeze(Response_time(m,:,:))./1000;

        % Fit LME model for each LBA model all over again...
        lme = LMEs{m};
        for subj=1:n_subj
            nexttile;
            hold on;
            for set_size_idx=1:length(Ns_values)
                plot(complexity(subj,set_size_idx), response_time(subj,set_size_idx), ".", "MarkerSize",18, "Color",cmap(set_size_idx,:))
            end
            tbl_new = table(repmat(subj,1,length(complexity_grid))',complexity_grid','VariableNames',{'Subject','PolicyComplexity'});
            RT_lme_theoretical = predict(lme, tbl_new);
            plot(complexity_grid, RT_lme_theoretical, "-", "LineWidth",1, 'Color',[0,0,0,0.1])
        
            xlim([0,log2(max(Ns_values))])
            ylim([0,min(2,max_RT)])
            if(mod(subj,n_subj/4)==1)
                ylabel("RT (s)")
            end
            if(subj>21)
                xlabel("Policy complexity (bits)")
            end
            title("Subject "+num2str(subj))
        end
        sgtitle(model_names(m)+": RT to policy complexity relationship")
    end
end

%% Helper functions for generating data from given set of parameters. 
function [datas] = fake_data_gen(experiment, datas, Params, model, Ns_values, Na, sv, m)
    n_subj = length(datas);
    switch m
        case 1
            Ncond=1;
        case 2
            Ncond=length(Ns_values);
        case 3
            Ncond=length(Ns_values);
    end
    
    true_Params = cell(n_subj,1);
    for subj=1:n_subj
        
        % Load true human data. 
        cond = datas{subj}.cond;
        stim = datas{subj}.stim;
        correct = datas{subj}.correct; % To be modified
        response = datas{subj}.response; % To be modified
        rt = datas{subj}.rt; % To be modified
    
        n_trials = length(cond);
    
        true_params = Params{subj}{m}; % Use these params as the generative params for fake data
        true_Params{subj} = true_params;
        [VS_correct A B VS_incorrect T0] = LBA_parse(model(m), true_params, Ncond);
        if(contains(experiment,"exp3"))
            switch m
                case 1
                    a = A; b = B+a; t0=T0;
                    vs_correct = repmat(VS_correct,1,length(Ns_values));
                    vs_incorrect = repmat(VS_incorrect,1,length(Ns_values));
                case 2
                    a=A(1); b=B(1)+a; t0=T0(1);
                    vs_correct = VS_correct;
                    vs_incorrect = VS_incorrect;
                case 3
                    vs_correct = VS_correct;
                    vs_incorrect = VS_incorrect;
            end
        elseif(contains(experiment,"exp1"))
            switch m
                case 1
                    a = A; b = B+a; t0=T0;
                    vs_correct = repmat(VS_correct,1,length(Ns_values));
                    vs_incorrect = repmat(VS_incorrect,1,length(Ns_values));
                case 2
                    a=A(1); t0=T0(1);
                    b_conds = B+a;
                    vs_correct = VS_correct;
                    vs_incorrect = VS_incorrect;
                case 3
                    vs_correct = VS_correct;
                    vs_incorrect = VS_incorrect;
            end

        end
    
        for t=1:n_trials
            %[subj,m,t]
            set_size_idx_trial = cond(t);
            v = repmat(vs_incorrect(set_size_idx_trial),1,Na);
            v(stim(t))=vs_correct(set_size_idx_trial);
            if(m==3)
                a=A(set_size_idx_trial); b=B(set_size_idx_trial)+a; t0=T0(set_size_idx_trial);
            elseif(contains(experiment,"exp1") && m==2)
                b = b_conds(set_size_idx_trial);
            end
            [datas{subj}.response(t), datas{subj}.rt(t), ~] = LBA_trial(a, b, v, t0, sv, Na);
            datas{subj}.correct(t) = double(datas{subj}.response(t)==stim(t));
        end
    end
end

function [] = true_fitted_choice_rt_histograms(Datas_orig, Datas_fitted, models_to_plot)
    %n_subj = length(Datas_orig{1});
    n_subj = 28;
    for m=models_to_plot
        figure
        tiledlayout(4,n_subj/4, 'Padding', 'none', 'TileSpacing', 'compact'); 
        for subj=1:n_subj
            nexttile(subj)
            hold on
            histogram(Datas_orig{m}{subj}.response)
            histogram(Datas_fitted{m}{subj}.response)
            xticks([1:6])
            xticklabels([1:6])
            title("Subject "+num2str(subj))
        end
        sgtitle("Model "+num2str(m)+": true and fitted choice")
        
        figure
        tiledlayout(4,n_subj/4, 'Padding', 'none', 'TileSpacing', 'compact'); 
        for subj=1:n_subj
            nexttile(subj)
            hold on
            truth_rts = Datas_orig{m}{subj}.rt;
            histogram(truth_rts, -100:100:2000)
            fitted_rts = Datas_fitted{m}{subj}.rt;
            histogram(fitted_rts,  -100:100:2000)
            title("Subject "+num2str(subj))
        end
        sgtitle("Model "+num2str(m)+": true and fitted RT")
    end
end
