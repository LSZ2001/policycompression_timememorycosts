function [params LL] = LBA_mle(data, model, pArray, Na)
% Continuous maximum likelihood estimation of LBA model
% [params fVal] = LBA_mle(data, model, pArray)
%
% Provides maximum likelihood fits to vector of choice and RT data. User
% can set various parameterisations of model via "model" structure
%
% Usage:
%
% Inputs:
%
% data --> structure containing following fields, all vectors of length
% 1 x trials
% (RT data and missed trials should be cleaned before passing to LBA_mle, see LBA_clean)
% data.rt - RT in milliseconds
% data.cond - condition vector (e.g. 1=easy, 2=hard)
% if no conditions, specify all as "1"
% data.stim - stimulus code (e.g. 1=left, 2=right)
% data.response - response code (e.g. 1=left, 2=right)
% data.coherence - optional, if present code will estimate link
% parameter from stimulus coherence into drift rate (cf. Palmer et al.,
% 2005, J Vis)
% All fields in data should be columnar vectors/matrices
%
% model --> structure containing information on which parameters to
% share between conditions. Must contain fields for v, A, b, sv, t0. Each
% field must be a scalar equal to either 1 or Ncond.
% E.g. to share bounds between 3 conditions, but to keep drift rates
% constant, set:
% model.v = 1; model.A = 1; model.b = 3; model.sv = 1; model.t0 = 1;
%
% pArray --> vector of starting points for parameters to be estimated
% pArray = [v A b-A t0 sv]
% length of pArray corresponds to setup in model
%
% Outputs:
%
% params --> vector of fitted parameters in same order as pArray (note
% output of b-A, need to subtract condition-specific A to get b)
%
% LL --> log-likelihood of data given model
%
% SF 2012 sf102@nyu.edu

options = optimset('Display','iter','MaxFunEvals',100000);
LB = ones(1,length(pArray)).*1e-5;
UB = ones(1,length(pArray)).*Inf;

% For p_lrate, their values must be in [0,1].
field_names = fieldnames(model);
if(field_names{end}=="p_lrate")
    p_lrate_nparams = model.p_lrate;
    UB((end-p_lrate_nparams+1):end) = 1-(1e-5);
end

[params fVal] = fmincon(@fitfunc,pArray,[],[],[],[],LB,UB,[],options);
%[params fVal] = bads(@fitfunc,pArray,LB,UB,LB,ones(1,length(pArray)).*1000);

LL = -fVal; % This returns the log likelihood. 

    function negLL = fitfunc(pArray)
        
        Ncond = max(data.cond);
        ntrials = length(data.response);
        
        % ensure data is in right format
        data = structfun(@(x) reshape(x,ntrials,1), data, 'UniformOutput', false);
        
        [v_correct A b v_incorrect t0, perserv, perservA, v_lrate, p_lrate] = LBA_parse(model, pArray, Ncond);
        A = real(log(A));
        b = real(log(b));
        t0 = real(log(t0));

        model_contains_perseveration_prob = ((1-isscalar(v_lrate)) + (1-isnan(v_lrate))) > 0;
        model_contains_perseveration_prob = model_contains_perseveration_prob(1);
        
        
        %% Get likelihoods
        if isfield(data, 'cond')
            
            % Get log-liks for these parameters
            cor = data.correct; %data.response == data.stim;
            if Ncond == 1
                vi = repmat(v_incorrect, length(data.cond), Na);
            else
                vi = repmat(v_incorrect(data.cond), 1, Na);
            end
            rtfit = data.rt - exp(t0(data.cond));
            sv = repmat(0.1, length(data.cond),1); % Parameter identifiability

            % NOTE: LBA_n1PDF(t, ..., v, ...) finds the prob that the
            % FIRST accumulator (corresponding to v(:,1) is the earliest
            % accumulator to reach the bound at time t.
            % HENCE: we need to ensure that v(trial,1)=v_correct if the
            % participant answers this trial correctly, and
            % v(trial,1)=v_incorrect otherwise (and set any
            % arbitrary col to v_correct instead--I have just chosen Col
            % 2).
            v_correct_ind = sub2ind([length(data.cond), Na], 1:length(data.cond), logical(~cor)'+1);
            vi(v_correct_ind) = v_correct(data.cond);

            if(model_contains_perseveration_prob)
                % NOTE: LBA_n1PDF(t, ..., v, ...) finds the prob that the
                % FIRST accumulator (corresponding to v(:,1) is the earliest
                % accumulator to reach the bound at time t.
                % ALSO: v(trial,1)=v_correct if the
                % participant answers this trial correctly, and
                % v(trial,1)=v_incorrect otherwise (and set any
                % arbitrary col to v_correct instead--I have just chosen Col
                % 2).
                perserv_accumulator_matrix = lba_fits_perserv_prob_accumulator(data, Na, v_lrate, p_lrate);
                vi = exp(vi+perserv_accumulator_matrix);
            end
        p = LBA_n1PDF(rtfit, exp(A(data.cond)), exp(b(data.cond)) + exp(A(data.cond)), vi, sv);
            
        else
            fprintf('\n\n\nBad input! See help LBA_mle.\n\n');
            return;
        end
        
        p(p<=1e-5) = 1e-5;  % avoid underflow
        negLL = -sum(log(p));
    end
end