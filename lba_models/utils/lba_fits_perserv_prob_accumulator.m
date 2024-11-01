function [perserv_accumulator_matrix] = lba_fits_perserv_prob_accumulator(data, Na, v_lrate, p_lrate, is_model_fit)
% This code is only used for LBA model fits (NLL evaluation). It cannot
% be used for generating fake data from an already-fitted LBA. 
    if(nargin==4)
        is_model_fit=true;
    end
    n_trials = length(data.cond);
    n_conds = 3;
    %n_conds = length(unique(data.cond));

    perserv_accumulator_matrix = zeros(n_trials, Na);
    P_a_init = ones(1,Na)./Na; % Initialize each condition with flat P(a)

    for c=1:n_conds
        P_a = P_a_init; % Initialize each condition with flat P(a)
        v_addon = v_lrate(c).*P_a; % Initialize each condition with no perseveration.

        % Perseveration does not happen at transition across conds.
        relevant_trials = find(data.cond==c);
        actions = data.response(relevant_trials);
        corrchoice = data.corrchoice(relevant_trials);

        perserv_accumulator_matrix(relevant_trials(1),:) = v_addon;
        for trial=2:length(relevant_trials)
            % Get the previous trial's chosen action of human.
            prevtrial_action = actions(trial-1);
            % Get the current trial's chosen action of human.
            thistrial_action = actions(trial);
            thistrial_corraction = corrchoice(trial);


            % thistrial_otheractions = setdiff(1:Na,[thistrial_action,thistrial_corraction]);
            % Create a logical mask for values to be excluded
            mask = true(1, Na);
            mask([thistrial_action,thistrial_corraction]) = false;
            % Extract the indices that are true
            % thistrial_otheractions = find(mask);


            % Update P(a) based on prev trial's choice, then normalize
            P_a_addon = ([1:Na]==prevtrial_action);
            %P_a(prevtrial_action) = P_a(prevtrial_action) + p_lrate(c); 
            P_a = P_a + p_lrate(c).*(P_a_addon-P_a);
            P_a = P_a ./sum(P_a);

            % Update v_addon based on new P(a), to sample the current
            % trial's action.
            %v_addon = v_lrate(c).*P_a;
            v_addon = v_lrate(c).*log(P_a);


            % Note: P_a, v_addon are constructed assuming action indexing used in the human
            % dataset.
            % But because the LBA model fit code assumes that the first
            % accumulator is the chosen action, we need to make changes.
            perserv_accumulator_matrix(relevant_trials(trial),1) = v_addon(thistrial_action);

            % Also in my LBA_mle code, if the chosen action is incorrect,
            % then the second action is the correct action.
            if(thistrial_action==thistrial_corraction)
                % perserv_accumulator_matrix(relevant_trials(trial),2:end) = v_addon(thistrial_otheractions);
                perserv_accumulator_matrix(relevant_trials(trial),2:end) = v_addon(mask);
            else
                perserv_accumulator_matrix(relevant_trials(trial),2) = v_addon(thistrial_corraction);
                %perserv_accumulator_matrix(relevant_trials(trial),3:Na) = v_addon(thistrial_otheractions);
                perserv_accumulator_matrix(relevant_trials(trial),3:Na) = v_addon(mask);
            end
            % perserv_accumulator_matrix(relevant_trials(trial),:) = v_addon;

            % These perserv_accumulator_matrix values are now used to
            % generate choice for the current trial.
        end
    end
end