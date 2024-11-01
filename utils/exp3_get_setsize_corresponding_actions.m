function [datas, correct_actions] = exp3_get_setsize_corresponding_actions(datas, set_sizes)
    % This code takes in the datas saved file for Experiment 3. Then for
    % each subject in each set size Ns = {2,4,6}, it adds an extra data column entry
    % that is a length-Ns vector. The ith entry of this vector denotes the
    % number key action that is optimal for state i. 
    if(nargin==1)
        set_sizes = [2,4,6];
    end

    n_subj = length(datas.test.("set_size_2"));
    correct_actions = zeros(n_subj,length(set_sizes), max(set_sizes));
    correct_actions(:) = NaN;
    for subj=1:n_subj
        for c=1:length(set_sizes)
            data_train_subj_cond = datas.train.("set_size_"+set_sizes(c))(subj);
            data_test_subj_cond = datas.test.("set_size_"+set_sizes(c))(subj);


            states = unique(data_test_subj_cond.s,'stable')';
            states_corresponding_actions = unique(data_test_subj_cond.corrchoice,'stable')';

            correct_action = zeros(1,max(states));
            correct_action(states) = states_corresponding_actions;
            datas.train.("set_size_"+set_sizes(c))(subj).correct_action  = correct_action;
            datas.test.("set_size_"+set_sizes(c))(subj).correct_action  = correct_action;
            correct_actions(subj,c,1:max(states)) = correct_action;

        end
    end
end