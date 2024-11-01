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
        corrchoice = [];
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
                data_task_subj.corrchoice = data_task_subj.corrchoice(idx);
            end
    
            num_trials_task_subj = length(data_task_subj.rt);
            rt=[rt; data_task_subj.rt];
            stim=[stim; data_task_subj.s];
            response=[response; data_task_subj.a];
            cond=[cond; c.*ones(num_trials_task_subj,1)];
            correct=[correct; data_task_subj.acc];
            corrchoice=[corrchoice; data_task_subj.corrchoice];
        end
        data.cond=cond;
        data.stim=stim;
        data.response=response;
        data.rt=rt;
        data.correct=correct;
        data.corrchoice=corrchoice;
        datas{subj} = data;
    end
end