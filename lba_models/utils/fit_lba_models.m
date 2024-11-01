
function [Params_best, Params, LLs, pArray] = fit_lba_models(experiment, Datas, model, n_inits, model_idxs, task_vals, Na, is_paramrecov)
    % Given an LBA-compatible dataset and model specifications (in models), fit the models with indexing model_idxs (either [1,2] or [3]).  
    if(is_paramrecov)
        n_subj = length(Datas{1});
        models = length(Datas);
    else
        n_subj = length(Datas);
        models = 4;
    end

    Nc = length(task_vals);
    % Note: model.sv is actually v_incorrect now!
    pArray = cell(models,n_inits);
    for init=1:n_inits
        pArray{1,init} = rand(1,5).*[2,500,200,2,0.1]; %[0.8 300 150 0.4 1];
        %LBA_parse takes in params in this sequence: [v_correct(s), A,
        %b(s), v_incorrect(s) (known as sv), t0, others]
        % LBA 2 varies the mean drift rates v_correct, v_incorrect.
        pArray{2,init} = rand(1,Nc+4).*[2,500,repmat(200,1,Nc),2,0.1];
        % LBA4 is same as LBA2, aside from adding cond-specific perseveration parameters.
        pArray{4,init} = rand(1,2*Nc+5).*[10,500,repmat(200,1,Nc),2,5, repmat(1,1,Nc),1];
        % LBA3 is fitting LBA1 parameters separately for each condition.
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

            parfor init=1:n_inits   %parfor init=1:n_inits
                [subj, m, init]
                params_Temp = zeros(1, length(pArray{3,1}));
                if(m~=3) % LBA 1, 2, and 4.
                    [params{m,init}, LL(m,init)] = LBA_mle(datas{subj}, model(m), pArray{m,init}, Na);
                else % LBA 3; Fit independent LBA for each ITI/set size condition
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
                        data.corrchoice = datas{subj}.corrchoice(relevant_trialidx_cond);
    
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

        save("lba_models\fits\lba_"+experiment+"_model"+m+"_temp.mat","pArray","Params","LLs","model_idxs")
    end

    % Find best fit across all inits for each subject and model. 
    % Note: must use fit_lba_models([3]) to fit LBA 3 separately from all
    % other models!!
    Params_best = cell(n_subj,1);
    if(~ismember(3, model_idxs)) % Use when model_idxs = [1,2]
        for subj=1:n_subj
            [~, loglike_max_init] = max(LLs{subj},[],2);
            params_best = cell(3,1);
            for m=model_idxs 
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


