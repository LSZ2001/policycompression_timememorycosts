function [model, pArray,names,Params,LLs, Datas, True_Params, Params_best,LLs_model3] = merge_lba_fits(experiment, lba_folder, is_paramrecov)
    % For the saved LBA fits for Model 1 to 4, merge their saved files.
    model2_fit_filename = "lba_"+experiment+"_model2";
    load(lba_folder+"fits/"+model2_fit_filename)
    LLs_model2 = LLs;
    Params_model2 = Params;
    pArray_model2 = pArray;

    model3_fit_filename = "lba_"+experiment+"_model3";
    load(lba_folder+"fits/"+model3_fit_filename)
    LLs_model3 = LLs;
    Params_model3 = Params;
    pArray_model3 = pArray;
    Params_best_model3 = Params_best;

    % Perseverative LBA4 
    model4_fit_filename = "lba_"+experiment+"_model4";
    load(lba_folder+"fits/"+model4_fit_filename)
    LLs_model4 = LLs;
    Params_model4 = Params;
    pArray_model4 = pArray;
    
    % LBA1
    model1_fit_filename = "lba_"+experiment+"_model1";
    load(lba_folder+"fits/"+model1_fit_filename)
    n_subj=length(LLs);
    n_inits=size(Params{1});
    n_inits = n_inits(2);
    for init=1:n_inits
        pArray{2,init} = pArray_model2{2,init};
        pArray{3,init} = pArray_model3{3,init};
        pArray{4,init} = pArray_model4{4,init};
    end
    for subj=1:n_subj
        for init=1:n_inits
            Params{subj}{2,init} = Params_model2{subj}{2,init};
            Params{subj}{3,init} = Params_model3{subj}{3,init};
            Params{subj}{4,init} = Params_model4{subj}{4,init};
        end
    end
    

    % In Model 3, the three conditions are fitted separately, hence we have
    % 3 LL values per subj*init. 
    % In the main LL struct, we will only store the sum of the 3 LLs. 
    % The actual 3 LLs for model 3 are stored separately in
    % LLs_model3_sep.
    n_subj = length(LLs);
    n_inits = size(LLs{1},2);
    for subj=1:n_subj
        for init=1:n_inits
            LLs{subj}(2,init)=LLs_model2{subj}(2,init);
            LLs{subj}(3,init)=sum(cell2mat(LLs_model3{subj}(3,init)));
            LLs{subj}(4,init)=LLs_model4{subj}(4,init);
        end
    end



    Params_best = cell(n_subj,1);
    for subj=1:n_subj
        [~, loglike_max_init] = max(LLs{subj},[],2);
        params_best = cell(3,1);
        for m=[1,2,4]
            params_best{m} = Params{subj}{m,loglike_max_init(m)};
        end
        params_best{3} = Params_best_model3{subj}{3};
        Params_best{subj} = params_best;
    end


    if(~is_paramrecov)
        True_Params = NaN;
        Datas = NaN;
    end
end