clear all; close all;
% base_folder = 'C:\Users\liu_s\policycompression_timememorycosts';
% cd(base_folder)
% addpath(genpath(base_folder))

% Figure and font default setting
set(0,'units','inches');
Inch_SS = get(0,'screensize');
set(0,'units','pixels');
figsize = get(0, 'ScreenSize');
Res = figsize(3)./Inch_SS(3);
set(groot, 'DefaultAxesTickDir', 'out', 'DefaultAxesTickDirMode', 'manual');
fontsize=12;
set(groot,'DefaultAxesFontName','Arial','DefaultAxesFontSize',fontsize);
set(groot,'DefaultLegendFontSize',fontsize-2,'DefaultLegendFontSizeMode','manual')

% paths
figformat = "svg";
figpath = "figures\"; %"newplots\"
datapath = "data\";
lba_folder = "lba_models\";
png_dpi = 500;

% Color palettes
cmap = brewermap(3, 'Set1');
cmap = cmap([1,3,2],:);
cmap_subj = brewermap(200, 'Set1');
cmap_exp3 = brewermap(9, 'Set2');
cmap_exp3 = cmap_exp3([2,1,3,4,5,6,9],:);

% Load data (test block data only)
load(datapath+"iti_bandit_data_exp1.mat");
data_exps.exp1 = datas.test;
survey_exps.exp1 = datas.survey;
load(datapath+"iti_bandit_data_exp2.mat");
data_exps.exp2 = datas.test;
survey_exps.exp2 = datas.survey;
load(datapath+"iti_bandit_data_exp3.mat");
data_exps.exp3 = datas.test;
survey_exps.exp3 = datas.survey;
load(datapath+"iti_bandit_data_exp3_old2.mat");
data_exps.exp3_old2 = datas.test;
survey_exps.exp3_old2 = datas.survey;
clear data datas;

%% Parse data
experiment = "exp1";
[exp1.mturkIDs, exp1.optimal_sol, exp1.BehavioralStats, exp1.LME, exp1.TTests, exp1.TTestsCI, exp1.CohensD, exp1.WilcoxonTests] = parse_data(data_exps, survey_exps, experiment);
experiment = "exp2";
[exp2.mturkIDs, exp2.optimal_sol, exp2.BehavioralStats, exp2.LME, exp2.TTests, exp2.TTestsCI, exp2.CohensD, exp2.WilcoxonTests] = parse_data(data_exps, survey_exps, experiment);
experiment = "exp3";
[exp3.mturkIDs, exp3.optimal_sol, exp3.BehavioralStats, exp3.LME, exp3.TTests, exp3.TTestsCI, exp3.CohensD, exp3.WilcoxonTests] = parse_data(data_exps, survey_exps, experiment);

%% Add in Experiment 1 and 3 LBA results
exp1 = append_lba_preds("exp1", exp1, lba_folder);
exp3 = append_lba_preds("exp3", exp3, lba_folder);

%%
exps.exp1 = exp1; exps.exp2 = exp2; exps.exp3 = exp3; 

%% Fig. 1 (excluding Figure 1A)
figspecs = [0 0 figsize(3)*0.6 figsize(4)];
Figure1(exps, figspecs, cmap, cmap_exp3);
saveas(gca, figpath+'Fig1_partial.fig')
exportgraphics(gcf,figpath+'Fig1_partial.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig1_partial.pdf',"ContentType","vector");

%% Fig. 2 (only including Figure 2B)
figspecs = [0 0 figsize(4)*0.4 figsize(4)*0.4];
Figure2(exp1, figspecs);
saveas(gca, figpath+'Fig2_partial.fig')
exportgraphics(gcf,figpath+'Fig2_partial.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig2_partial.pdf',"ContentType","vector");

%% Fig. 3
figspecs = [0 100 figsize(3) figsize(3)*0.35];
Figure3(exp1,cmap, figspecs);
saveas(gca, figpath+'Fig3.fig')
exportgraphics(gcf,figpath+'Fig3.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig3.pdf',"ContentType","vector");

%% Fig. 4
figspecs = [0 0 figsize(3) figsize(4)];
Figure4(exp1, cmap, figspecs);
saveas(gca, figpath+'Fig4.fig')
exportgraphics(gcf,figpath+'Fig4.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig4.pdf',"ContentType","vector");

%% Fig. 5
figspecs = [0 0 min(figsize(3:4))*2 min(figsize(3:4))./2.5];
Figure5(exp2, cmap, figspecs)
saveas(gca, figpath+'Fig5_partial.fig')
exportgraphics(gcf,figpath+'Fig5_partial.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig5_partial.pdf',"ContentType","vector");

%% Fig. 6
figspecs = [0 100 figsize(3) figsize(3)*0.35];
Figure6(exp3,cmap_exp3, figspecs);
saveas(gca, figpath+'Fig6.fig')
exportgraphics(gcf,figpath+'Fig6.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'Fig6.pdf',"ContentType","vector");

%% Figure S1: Subjects did not exploit the counterbalanced cycles
figspecs = [0 0 figsize(3)*0.37 figsize(4)];
experiment_names = ["exp1","exp2","exp3"];
FigureS1(exps, experiment_names, figspecs, cmap, cmap_exp3)
saveas(gca, figpath+'FigS1.fig')
exportgraphics(gcf,figpath+'FigS1.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS1.pdf',"ContentType","vector");

%% Figure S2: even subject with low policy complexity are adjusting it across conditions
figspecs = [0 0 figsize(3) figsize(4)];
experiment_names = ["exp1","exp2","exp3"];
FigureS2(exps, experiment_names, figspecs)
saveas(gca, figpath+'FigS2.fig')
exportgraphics(gcf,figpath+'FigS2.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS2.pdf',"ContentType","vector");

%% Figure S3: Complexity to RT relationship, and all LME fit plots.
figspecs = [0 0 figsize(3) figsize(4)*0.85];
FigureS3(exps, cmap, cmap_exp3, figspecs)
saveas(gca, figpath+'FigS3_partial.fig')
exportgraphics(gcf,figpath+'FigS3_partial.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS3_partial.pdf',"ContentType","vector");


%% Figure S4: LBA results for Experiment 1 and 3. 
figspecs = [0 0 figsize(3)*0.72 figsize(4)*0.8];
FigureS4([1,3], exps, cmap_exp3([1,2,3,4,7,6],:), cmap_exp3, figspecs)
saveas(gca, figpath+'FigS4.fig')
exportgraphics(gcf,figpath+'FigS4.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS4.pdf',"ContentType","vector");


%% Figure S5: 5 set-size conditions
figspecs = [0 0 min(figsize(3:4))*1.4 min(figsize(3:4))*0.8];
[exp3_old2.mturkIDs, exp3_old2.optimal_sol, exp3_old2.BehavioralStats, exp3_old2.LME, exp3_old2.TTests, exp3_old2.WilcoxonTests] = FigureS5("exp3_old2", data_exps.exp3_old2, survey_exps.exp3_old2, cmap_exp3([4,1,5,2,3],:), figspecs);
saveas(gca, figpath+'FigS5.fig')
exportgraphics(gcf,figpath+'FigS5.png','Resolution',png_dpi);
exportgraphics(gcf,figpath+'FigS5.pdf',"ContentType","vector");




%% Cohen's d function
extractFirstColumn(exp1.CohensD)
function outputStruct = extractFirstColumn(inputStruct)
    % Initialize the output struct
    outputStruct = struct();
    
    % Get the fieldnames of the input struct
    fields = fieldnames(inputStruct);
    
    % Loop through each field
    for i = 1:numel(fields)
        fieldName = fields{i};
        % Extract the first column of the 3x2 cell and convert it to a 1x3 vector
        outputStruct.(fieldName) = cell2mat(inputStruct.(fieldName)(:, 1))';
    end
end


%% Helper functions
function [mturkIDs, optimal_sol, BehavioralStats, LME, TTests, TTestsCI, CohensD, WilcoxonTests] = parse_data(data_exps, survey_exps, experiment)
    TTests = struct();
    CohensD = struct();
    WilcoxonTests = struct();

    data = data_exps.(experiment);
    survey = survey_exps.(experiment);
    feedback_duration = 0.3; % in seconds
    
    switch experiment
        case "exp1"
            Q = eye(4).*0.5+0.25;
            n_subj = length(data);
            cond = unique(data(1).cond); % ITI conditions
        case "exp2"
            Q = [0.75 0.25 0.25 0.25; 0.75 0.25 0.25 0.25; 0.25 0.25 0.75 0.25; 0.25 0.25 0.25 0.75];
            n_subj = length(data);
            cond = unique(data(1).cond); % ITI conditions
        case "exp3"
            set_sizes = [2,4,6];
            cond = [2000];
            n_tasks = length(set_sizes);
            task_names = "set_size_"+set_sizes;
            datas = data;
            n_subj = length(data.(task_names(1)));
    end

    reward_count = zeros(n_subj,3);
    cond_entropy = zeros(n_subj,3); % H(A|S)
    repeat_actions = zeros(n_subj,3); % Measure of perserverance
    mturkIDs = [];
    if(experiment ~= "exp3")
        for s = 1:n_subj
            mturkIDs = [mturkIDs; convertCharsToStrings(data(s).ID)];
            for c = 1:length(cond)
                idx = data(s).cond == cond(c);
                state = data(s).s(idx);
                action = data(s).a(idx);
                acc = data(s).acc(idx);
                r = data(s).r(idx);
                rt = data(s).rt(idx);
                tt = data(s).tt(idx); % total time of the block
                n_trials(s,c) = length(state);
                accuracy(s,c) = nanmean(acc);
                reward(s,c) = nanmean(r);
                reward_count(s,c) = sum(r);
                reward_rate(s,c) = reward_count(s,c)/tt(end);
                complexity(s,c) = mutual_information(round(state),round(action),0.1)./log(2);
                response_time(s,c) = nanmean(rt./1000); % RT in seconds
                cond_entropy(s,c) = condEntropy(round(action), round(state));
                repeat_actions(s,c) = nanmean(action(1:end-1) == action(2:end));
                first_cond(s) = data(s).first_cond;
            end
        end
    else % Experiment 3
        for task = 1:n_tasks
            task_name = task_names(task);
            data = datas.(task_name);
            for s = 1:n_subj
                if(task==1)
                    mturkIDs = [mturkIDs; convertCharsToStrings(data(s).ID)];
                end
                iti_2_idx = data(s).cond == cond(end);
                for c = 1:length(cond)
                    idx = data(s).cond == cond(c);
                    state = data(s).s(idx);
                    action = data(s).a(idx);
                    acc = data(s).acc(idx);
                    r = data(s).r(idx);
                    rt = data(s).rt(idx);
                    tt = data(s).tt(idx); % total time of the block
        
                    % Dimensions are: [subject, set size].
                    n_trials(s,task) = length(state);
                    accuracy(s,task) = nanmean(acc);
                    reward(s,task) = nanmean(r);
                    reward_count(s,task) = sum(r);
                    reward_rate(s,task) = reward_count(s,task)/tt(end);
                    complexity(s,task) = mutual_information(round(state),round(action),0.1)./log(2); 
                    response_time(s,task) = nanmean(rt./1000); % RT in seconds
                    cond_entropy(s,task) = condEntropy(round(action), round(state));
                    repeat_actions(s,task) = nanmean(action(1:end-1) == action(2:end));
                end
            end
        end
    end
    BehavioralStats.n_trials = n_trials;
    BehavioralStats.accuracy=accuracy;
    BehavioralStats.reward=reward;
    BehavioralStats.reward_rate=reward_rate;
    BehavioralStats.complexity=complexity;
    BehavioralStats.response_time=response_time;
    BehavioralStats.cond_entropy=cond_entropy;    
    BehavioralStats.repeat_actions=repeat_actions;

    
    % Compute Cohen's D
    CohensD_complexity = cell(3,2); % ITI_cond
    CohensD_response_time = cell(3,2); % ITI_cond
    CohensD_perc_reward = cell(3,2); % ITI_cond
    CohensD_reward_rate = cell(3,2); % ITI_cond
    CohensD_cond_entropy = cell(3,2);
    CohensD_repeat_actions = cell(3,2);
    ITI_pairs = [1,2; 2,3; 1,3];
    for iti_idx=1:length(ITI_pairs)
        CohensD_complexity(iti_idx,:) = table2cell(meanEffectSize(complexity(:,ITI_pairs(iti_idx,1)),complexity(:,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
        CohensD_response_time(iti_idx,:) = table2cell(meanEffectSize(response_time(:,ITI_pairs(iti_idx,1)),response_time(:,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
        CohensD_reward_rate(iti_idx,:) = table2cell(meanEffectSize(reward_rate(:,ITI_pairs(iti_idx,1)),reward_rate(:,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
        CohensD_cond_entropy(iti_idx,:) = table2cell(meanEffectSize(cond_entropy(:,ITI_pairs(iti_idx,1)),cond_entropy(:,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
        CohensD_repeat_actions(iti_idx,:) = table2cell(meanEffectSize(repeat_actions(:,ITI_pairs(iti_idx,1)),repeat_actions(:,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
    end
    CohensD.complexity = CohensD_complexity;
    CohensD.response_time = CohensD_response_time;
    CohensD.cond_entropy = CohensD_cond_entropy;
    CohensD.repeat_actions = CohensD_repeat_actions;
    CohensD.reward_rate = CohensD_reward_rate;
    
    [~,TTests.complexity(1),TTestsCI.complexity(:,1),~] = ttest(complexity(:,1), complexity(:,2), "Tail","left");
    [~,TTests.complexity(2),TTestsCI.complexity(:,2),~] = ttest(complexity(:,2), complexity(:,3), "Tail","left");
    [~,TTests.complexity(3),TTestsCI.complexity(:,3),~] = ttest(complexity(:,1), complexity(:,3), "Tail","left");
    [~,TTests.response_time(1),TTestsCI.response_time(:,1),~] = ttest(response_time(:,1), response_time(:,2), "Tail","left");
    [~,TTests.response_time(2),TTestsCI.response_time(:,2),~] = ttest(response_time(:,2), response_time(:,3), "Tail","left");
    [~,TTests.response_time(3),TTestsCI.response_time(:,3),~] = ttest(response_time(:,1), response_time(:,3), "Tail","left");
    [~,TTests.cond_entropy(1),TTestsCI.cond_entropy(:,1),~] = ttest(cond_entropy(:,2), cond_entropy(:,1), "Tail","left");
    [~,TTests.cond_entropy(2),TTestsCI.cond_entropy(:,2),~] = ttest(cond_entropy(:,3), cond_entropy(:,2), "Tail","left");
    [~,TTests.cond_entropy(3),TTestsCI.cond_entropy(:,3),~] = ttest(cond_entropy(:,3), cond_entropy(:,1), "Tail","left");
    [~,TTests.repeat_actions(1),TTestsCI.repeat_actions(:,1),~] = ttest(repeat_actions(:,2), repeat_actions(:,1), "Tail","left");
    [~,TTests.repeat_actions(2),TTestsCI.repeat_actions(:,2),~] = ttest(repeat_actions(:,3), repeat_actions(:,2), "Tail","left");
    [~,TTests.repeat_actions(3),TTestsCI.repeat_actions(:,3),~] = ttest(repeat_actions(:,3), repeat_actions(:,1), "Tail","left");
    [~,TTests.reward_rate(1),TTestsCI.reward_rate(:,1),~] = ttest(reward_rate(:,2), reward_rate(:,1), "Tail","left");
    [~,TTests.reward_rate(2),TTestsCI.reward_rate(:,2),~] = ttest(reward_rate(:,3), reward_rate(:,2), "Tail","left");
    [~,TTests.reward_rate(3),TTestsCI.reward_rate(:,3),~] = ttest(reward_rate(:,3), reward_rate(:,1), "Tail","left");
    
    WilcoxonTests.complexity(1) = signrank(complexity(:,1), complexity(:,2), "Tail","left");
    WilcoxonTests.complexity(2) = signrank(complexity(:,2), complexity(:,3), "Tail","left");
    WilcoxonTests.complexity(3) = signrank(complexity(:,1), complexity(:,3), "Tail","left");
    WilcoxonTests.response_time(1) = signrank(response_time(:,1), response_time(:,2), "Tail","left");
    WilcoxonTests.response_time(2) = signrank(response_time(:,2), response_time(:,3), "Tail","left");
    WilcoxonTests.response_time(3) = signrank(response_time(:,1), response_time(:,3), "Tail","left");
    WilcoxonTests.cond_entropy(1) = signrank(cond_entropy(:,2), cond_entropy(:,1), "Tail","left");
    WilcoxonTests.cond_entropy(2) = signrank(cond_entropy(:,3), cond_entropy(:,2), "Tail","left");
    WilcoxonTests.cond_entropy(3) = signrank(cond_entropy(:,3), cond_entropy(:,1), "Tail","left");
    WilcoxonTests.repeat_actions(1) = signrank(repeat_actions(:,2), repeat_actions(:,1), "Tail","left");
    WilcoxonTests.repeat_actions(2) = signrank(repeat_actions(:,3), repeat_actions(:,2), "Tail","left");
    WilcoxonTests.repeat_actions(3) = signrank(repeat_actions(:,3), repeat_actions(:,1), "Tail","left");
    WilcoxonTests.reward_rate(1) = signrank(reward_rate(:,2), reward_rate(:,1), "Tail","left");
    WilcoxonTests.reward_rate(2) = signrank(reward_rate(:,3), reward_rate(:,2), "Tail","left");
    WilcoxonTests.reward_rate(3) = signrank(reward_rate(:,3), reward_rate(:,1), "Tail","left");
    
    % Perceived difficulty
    T0 = table(mturkIDs,complexity,'VariableNames',["ID","policy_complexity"]);
    T1 = struct2table(survey); 
    T1 = table(table2array(T1(:,1)),table2array(T1(:,4)),'VariableNames',["ID","difficulty"]);
    T = innerjoin(T0,T1);
    if(experiment=="exp1")
        difficulties = cell2mat(table2array(T(:,3)));
    else
        difficulties = table2array(T(:,3));
    end
    BehavioralStats.difficulties=difficulties;
    [h,TTests.difficulty(1),TTestsCI.difficulty(:,1),~] = ttest(difficulties(:,1), difficulties(:,2), "Tail","left");
    [h,TTests.difficulty(2),TTestsCI.difficulty(:,2),~] = ttest(difficulties(:,2), difficulties(:,3), "Tail","left");
    [h,TTests.difficulty(3),TTestsCI.difficulty(:,3),~] = ttest(difficulties(:,1), difficulties(:,3), "Tail","left");
    WilcoxonTests.difficulty(1) = signrank(difficulties(:,1), difficulties(:,2), "Tail","left");
    WilcoxonTests.difficulty(2) = signrank(difficulties(:,2), difficulties(:,3), "Tail","left");
    WilcoxonTests.difficulty(3) = signrank(difficulties(:,1), difficulties(:,3), "Tail","left");
    for iti_idx=1:length(ITI_pairs)
        CohensD_difficulties(iti_idx,:) = table2cell(meanEffectSize(difficulties(:,ITI_pairs(iti_idx,1)),difficulties(:,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
    end
    CohensD.difficulties=CohensD_difficulties;

    
    % Theoretical curves assuming linear RTH
    n_tot = 50;
    beta_set = linspace(0.1,15,n_tot);
    if(experiment ~= "exp3")
        p_state = [0.25 0.25 0.25 0.25];
        optimal_sol.Q = Q;
        optimal_sol.p_state = p_state;
        [optimal_sol.R, optimal_sol.V, optimal_sol.Pa, optimal_sol.optimal_policy] = blahut_arimoto(p_state,Q,beta_set);
        R = optimal_sol.R; V=optimal_sol.V;

            % P(A|S) for Experiment 2
            P_a_given_s = zeros(n_subj,4,length(cond),4); % subj, states, conds, actions
            for subj=1:n_subj
                for state=1:4
                    s_idx = find(data(subj).s==state);
                    subj_thisstateoccurrences = data(subj).s(s_idx);
                    subj_actions_giventhisstate = data(subj).a(s_idx);
                    for c = 1:length(cond)
                        cond_idx = find(data(subj).cond(s_idx) == cond(c));
                        states = subj_thisstateoccurrences(cond_idx);
                        actions = subj_actions_giventhisstate(cond_idx);
                        [N,~] = histcounts(actions,0.5:1:4.5);
                        P_a_given_s(subj, state, c, :) = N./sum(N);
                    end
                end
            end
            BehavioralStats.P_a_given_s = P_a_given_s;

    else
        P_a_given_s = nan(n_subj,max(set_sizes),length(cond),max(set_sizes)); % subj, states, conds, actions
        Q_full = normalize(eye(max(set_sizes)), 'range', [0.25 0.75]);
        
        for set_size_idx = 1:length(set_sizes)
            set_size = set_sizes(set_size_idx);
            task_name = task_names(set_size_idx);
            p_state = ones(1,set_size)./set_size;
            p_states.(task_name) = p_state;
            Q = Q_full(1:set_size,:);
            Qs.(task_name) = Q;
            % initialize variables
            [R.(task_name),V.(task_name),Pa.(task_name), optimal_policy.(task_name)] = blahut_arimoto(p_state,Q,beta_set);
            % P(A|S) for Experiment 3
            data = datas.(task_name);
            for subj=1:n_subj
                for state=1:set_size
                    s_idx = find(data(subj).s==state);
                    actions = data(subj).a(s_idx);
                    [N,~] = histcounts(actions,0.5:1:(max(set_sizes)+.5));
                    P_a_given_s(subj, state, set_size_idx, :) = N./sum(N);
                end
            end
        end
        optimal_sol.Q = Qs; optimal_sol.p_state = p_states; optimal_sol.R = R; optimal_sol.V = V; optimal_sol.Pa = Pa; optimal_sol.optimal_policy = optimal_policy;
        BehavioralStats.P_a_given_s = P_a_given_s;
    end


    if(experiment=="exp2")
        for s=1:n_subj
            % [Subject, iti_condition].
            a_perserv(s) = mode(data(s).correct_action, 2);
            for c=1:length(cond)
                % [Subject, iti_condition, the 2 states where a_perserv is suboptimal].
                states_where_a_perserv_suboptimal(s,c,:) = find(data(s).correct_action~=a_perserv(s));
                P_a_perserv_given_suboptimal_s(s,c,:) =  P_a_given_s(s,states_where_a_perserv_suboptimal(s,c,:),c,a_perserv(s));
                P_a_perserv_given_optimal_s(s,c,:) =  P_a_given_s(s,setdiff(1:4, squeeze(states_where_a_perserv_suboptimal(s,c,:))),c,a_perserv(s));
            end
        end
        % Average over p(a_perserv | s) for the two states s
        % where a_perserv is suboptimal. 
        P_a_perserv_given_suboptimal_s_mean = mean(P_a_perserv_given_suboptimal_s,3);
        BehavioralStats.SuboptimalA = P_a_perserv_given_suboptimal_s_mean;
        for iti_idx=1:length(ITI_pairs)
            CohensD_SuboptimalA(iti_idx,:) = table2cell(meanEffectSize(P_a_perserv_given_suboptimal_s_mean(:,ITI_pairs(iti_idx,1)), P_a_perserv_given_suboptimal_s_mean(:,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
        end
        CohensD.SuboptimalA = CohensD_SuboptimalA;
        [~,TTests.SuboptimalA(1),TTestsCI.SuboptimalA(:,1),~] = ttest(P_a_perserv_given_suboptimal_s_mean(:,2), P_a_perserv_given_suboptimal_s_mean(:,1), "Tail","left");
        [~,TTests.SuboptimalA(2),TTestsCI.SuboptimalA(:,2),~] = ttest(P_a_perserv_given_suboptimal_s_mean(:,3), P_a_perserv_given_suboptimal_s_mean(:,2), "Tail","left");
        [~,TTests.SuboptimalA(3),TTestsCI.SuboptimalA(:,3),~] = ttest(P_a_perserv_given_suboptimal_s_mean(:,3), P_a_perserv_given_suboptimal_s_mean(:,1), "Tail","left");
        WilcoxonTests.SuboptimalA(1) = signrank(P_a_perserv_given_suboptimal_s_mean(:,2),P_a_perserv_given_suboptimal_s_mean(:,1), "Tail","left");
        WilcoxonTests.SuboptimalA(2) = signrank(P_a_perserv_given_suboptimal_s_mean(:,3),P_a_perserv_given_suboptimal_s_mean(:,2), "Tail","left");
        WilcoxonTests.SuboptimalA(3) = signrank(P_a_perserv_given_suboptimal_s_mean(:,3),P_a_perserv_given_suboptimal_s_mean(:,1), "Tail","left");
    end
    %% LME fitting

    % Flattening arrays
    complexity_flat = complexity(:); 
    response_time_flat = response_time(:);
    [complexity_sorted, complexity_sorted_idx] = sort(complexity_flat);
    response_time_sorted = response_time_flat(complexity_sorted_idx);
    LME.complexity_sorted = complexity_sorted;
    LME.response_time_sorted = response_time_sorted;
    LME.complexity_sorted_idx = complexity_sorted_idx;
    if(experiment~="exp3")
        subject_id = repmat(1:n_subj, 1, length(cond))';
    else
        subject_id = repmat(1:n_subj, 1, length(set_sizes))';
    end

    tbl = table(subject_id,complexity_flat,response_time_flat,'VariableNames',{'Subject','PolicyComplexity','RT'});
    lme = fitlme(tbl,'RT ~ PolicyComplexity + (1|Subject) + (PolicyComplexity-1|Subject)');
    LME.lme = lme;
    LME.tbl = tbl;

    % STD of random effects
    [~,~,stats] = randomEffects(lme,'Alpha',0.01);
    q = dataset2cell(stats(1:n_subj,4));
    w = dataset2cell(stats((n_subj+1):end,4));
    LME.random_effects_intercept_std = std(cell2mat(q(2:end)));
    LME.random_effects_complexity_std = std(cell2mat(w(2:end)));
    
    % RT_predictions vs. RT_true
    RT_lme = predict(lme); % Return 1SD, instead of 95% CI. 
    RT_lme_sorted = RT_lme(complexity_sorted_idx);
    LME.RT_lme = RT_lme;
    LME.RT_lme_sorted = RT_lme_sorted;

    % Counterfactual RT at different policy complexity levels
    if(experiment ~= "exp3")
        complexity_rrmax = zeros(n_subj, length(cond));
        RT_lme_theoretical = zeros(n_subj, length(R));
        for subj=1:n_subj
            tbl_new = table(repmat(subj,1,length(R))',R,'VariableNames',{'Subject','PolicyComplexity'});
            rt_lme_theoretical = predict(lme, tbl_new);
            RT_lme_theoretical(subj,:)=rt_lme_theoretical;
            for c=1:length(cond)
                rr = V ./ (rt_lme_theoretical + cond(c)/1000 + feedback_duration);
                [max_rr, max_rr_complexity] = max(rr);
                complexity_rrmax(subj,c) = R(max_rr_complexity);
            end
        end
        LME.RT_lme_theoretical = RT_lme_theoretical;
    else
        complexity_rrmax = zeros(n_subj, length(set_sizes));
        RT_lme_theoreticals = zeros(n_subj, length(R.(task_names(end))));
        for subj=1:n_subj
            for set_size_idx=1:length(set_sizes)
                set_size = set_sizes(set_size_idx);
                task_name = task_names(set_size_idx);
                tbl_new = table(repmat(subj,1,length(R.(task_name)))',R.(task_name),'VariableNames',{'Subject','PolicyComplexity'});
                RT_lme_theoretical = predict(lme, tbl_new);
                if(set_size_idx == length(set_sizes))
                    RT_lme_theoreticals(subj,:)=RT_lme_theoretical;
                end
                rr = V.(task_name) ./ (RT_lme_theoretical + cond/1000 + feedback_duration);
                [max_rr, max_rr_complexity] = max(rr);
                complexity_rrmax(subj,set_size_idx) = R.(task_name)(max_rr_complexity);
            end
        end
        LME.RT_lme_theoretical = RT_lme_theoreticals;
    end
    LME.complexity_rrmax = complexity_rrmax;
    rhos_subj = zeros(n_subj,1);
    for subj=1:n_subj
        rhos_subj(subj) = corr(complexity(subj,:)', complexity_rrmax(subj,:)',"Type", "Spearman");
    end
    BehavioralStats.complexity_difficulty_spearman = rhos_subj;
    CohensD.complexity_difficulty_spearman_ispositive = table2cell(meanEffectSize(rhos_subj,Effect="cohen"));
    [~,TTests.complexity_difficulty_spearman_ispositive,TTestsCI.complexity_difficulty_spearman_ispositive,~] = ttest(rhos_subj);
    WilcoxonTests.complexity_difficulty_spearman_ispositive = signrank(rhos_subj);
        

    % Leftward complexity bias
    complexity_diff_from_rrmax = complexity-complexity_rrmax;
    LME.complexity_diff_from_rrmax=complexity_diff_from_rrmax;
    for c=1:3
        CohensD_complexity_lessthan_rrmax(c,:) = table2cell(meanEffectSize(complexity(:,c),complexity_rrmax(:,c),Effect="cohen", Paired=true));
    end
    CohensD.complexity_lessthan_rrmax = CohensD_complexity_lessthan_rrmax;
    [~,TTests.complexity_lessthan_rrmax(1),TTestsCI.complexity_lessthan_rrmax(:,1)] = ttest(complexity(:,1), complexity_rrmax(:,1), "Tail","left");
    [~,TTests.complexity_lessthan_rrmax(2),TTestsCI.complexity_lessthan_rrmax(:,2)] = ttest(complexity(:,2), complexity_rrmax(:,2), "Tail","left");
    [~,TTests.complexity_lessthan_rrmax(3),TTestsCI.complexity_lessthan_rrmax(:,3)] = ttest(complexity(:,3), complexity_rrmax(:,3), "Tail","left");
    WilcoxonTests.complexity_lessthan_rrmax(1) = signrank(complexity(:,1), complexity_rrmax(:,1),"Tail","left");
    WilcoxonTests.complexity_lessthan_rrmax(2) = signrank(complexity(:,2), complexity_rrmax(:,2),"Tail","left");
    WilcoxonTests.complexity_lessthan_rrmax(3) = signrank(complexity(:,3), complexity_rrmax(:,3),"Tail","left");

    % Two subgroups analysis of RT and policy complexity
    switch experiment
        case "exp1"
            thres = -1;
            % Use ITI=2 to separate into 2 groups
            lowcomplexity_group_subjidx = find(squeeze(complexity(:,3)-complexity_rrmax(:,3))<thres);
            highcomplexity_group_subjidx = find(squeeze(complexity(:,3)-complexity_rrmax(:,3))>=thres);
        case "exp2"
            thres = 0.25;
            % Use ITI=9.5 to separate into 2 groups
            lowcomplexity_group_subjidx = find(squeeze(complexity(:,2)-complexity_rrmax(:,2))<thres);
            highcomplexity_group_subjidx = find(squeeze(complexity(:,2)-complexity_rrmax(:,2))>=thres);
        case "exp3"
            thres = -0.9;
            % Use ITI=9.5 to separate into 2 groups
            lowcomplexity_group_subjidx = find(squeeze(complexity(:,3)-complexity_rrmax(:,3))<thres);
            highcomplexity_group_subjidx = find(squeeze(complexity(:,3)-complexity_rrmax(:,3))>=thres);

    end
    complexity_group_subjidx = {lowcomplexity_group_subjidx, highcomplexity_group_subjidx};
    for iti_idx=1:length(ITI_pairs)
        CohensD_complexity_low_complexity(iti_idx,:) = table2cell(meanEffectSize(complexity(lowcomplexity_group_subjidx,ITI_pairs(iti_idx,1)),complexity(lowcomplexity_group_subjidx,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
        CohensD_response_time_low_complexity(iti_idx,:) = table2cell(meanEffectSize(response_time(lowcomplexity_group_subjidx,ITI_pairs(iti_idx,1)),response_time(lowcomplexity_group_subjidx,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
        CohensD_complexity_high_complexity(iti_idx,:) = table2cell(meanEffectSize(complexity(highcomplexity_group_subjidx,ITI_pairs(iti_idx,1)),complexity(highcomplexity_group_subjidx,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
        CohensD_response_time_high_complexity(iti_idx,:) = table2cell(meanEffectSize(response_time(highcomplexity_group_subjidx,ITI_pairs(iti_idx,1)),response_time(highcomplexity_group_subjidx,ITI_pairs(iti_idx,2)),Effect="cohen", Paired=true));
    end
    CohensD.complexity_low_complexity = CohensD_complexity_low_complexity;
    CohensD.response_time_low_complexity = CohensD_response_time_low_complexity;
    CohensD.complexity_high_complexity = CohensD_complexity_high_complexity;
    CohensD.response_time_high_complexity = CohensD_response_time_high_complexity;

    [~,TTests.complexity_lowcomplexity(1),TTestsCI.complexity_lowcomplexity(:,1)] = ttest(complexity(lowcomplexity_group_subjidx,1), complexity(lowcomplexity_group_subjidx,2),"Tail","left");
    [~,TTests.complexity_lowcomplexity(2),TTestsCI.complexity_lowcomplexity(:,2)] = ttest(complexity(lowcomplexity_group_subjidx,2), complexity(lowcomplexity_group_subjidx,3),"Tail","left");
    [~,TTests.complexity_lowcomplexity(3),TTestsCI.complexity_lowcomplexity(:,3)] = ttest(complexity(lowcomplexity_group_subjidx,1), complexity(lowcomplexity_group_subjidx,3),"Tail","left");
    [~,TTests.response_time_lowcomplexity(1),TTestsCI.response_time_lowcomplexity(:,1)] = ttest(response_time(lowcomplexity_group_subjidx,1), response_time(lowcomplexity_group_subjidx,2),"Tail","left");
    [~,TTests.response_time_lowcomplexity(2),TTestsCI.response_time_lowcomplexity(:,2)] = ttest(response_time(lowcomplexity_group_subjidx,2), response_time(lowcomplexity_group_subjidx,3),"Tail","left");
    [~,TTests.response_time_lowcomplexity(3),TTestsCI.response_time_lowcomplexity(:,3)] = ttest(response_time(lowcomplexity_group_subjidx,1), response_time(lowcomplexity_group_subjidx,3),"Tail","left");
    WilcoxonTests.complexity_lowcomplexity(1) = signrank(complexity(lowcomplexity_group_subjidx,1), complexity(lowcomplexity_group_subjidx,2),"Tail","left");
    WilcoxonTests.complexity_lowcomplexity(2) = signrank(complexity(lowcomplexity_group_subjidx,2), complexity(lowcomplexity_group_subjidx,3),"Tail","left");
    WilcoxonTests.complexity_lowcomplexity(3) = signrank(complexity(lowcomplexity_group_subjidx,1), complexity(lowcomplexity_group_subjidx,3),"Tail","left");
    WilcoxonTests.response_time_lowcomplexity(1) = signrank(response_time(lowcomplexity_group_subjidx,1), response_time(lowcomplexity_group_subjidx,2),"Tail","left");
    WilcoxonTests.response_time_lowcomplexity(2) = signrank(response_time(lowcomplexity_group_subjidx,2), response_time(lowcomplexity_group_subjidx,3),"Tail","left");
    WilcoxonTests.response_time_lowcomplexity(3) = signrank(response_time(lowcomplexity_group_subjidx,1), response_time(lowcomplexity_group_subjidx,3),"Tail","left");

    [~,TTests.complexity_highcomplexity(1),TTestsCI.complexity_highcomplexity(:,1)] = ttest(complexity(highcomplexity_group_subjidx,1), complexity(highcomplexity_group_subjidx,2),"Tail","left");
    [~,TTests.complexity_highcomplexity(2),TTestsCI.complexity_highcomplexity(:,2)] = ttest(complexity(highcomplexity_group_subjidx,2), complexity(highcomplexity_group_subjidx,3),"Tail","left");
    [~,TTests.complexity_highcomplexity(3),TTestsCI.complexity_highcomplexity(:,3)] = ttest(complexity(highcomplexity_group_subjidx,1), complexity(highcomplexity_group_subjidx,3),"Tail","left");
    [~,TTests.response_time_highcomplexity(1),TTestsCI.response_time_highcomplexity(:,1)] = ttest(response_time(highcomplexity_group_subjidx,1), response_time(highcomplexity_group_subjidx,2),"Tail","left");
    [~,TTests.response_time_highcomplexity(2),TTestsCI.response_time_highcomplexity(:,2)] = ttest(response_time(highcomplexity_group_subjidx,2), response_time(highcomplexity_group_subjidx,3),"Tail","left");
    [~,TTests.response_time_highcomplexity(3),TTestsCI.response_time_highcomplexity(:,3)] = ttest(response_time(highcomplexity_group_subjidx,1), response_time(highcomplexity_group_subjidx,3),"Tail","left");
    WilcoxonTests.complexity_highcomplexity(1) = signrank(complexity(highcomplexity_group_subjidx,1), complexity(highcomplexity_group_subjidx,2),"Tail","left");
    WilcoxonTests.complexity_highcomplexity(2) = signrank(complexity(highcomplexity_group_subjidx,2), complexity(highcomplexity_group_subjidx,3),"Tail","left");
    WilcoxonTests.complexity_highcomplexity(3) = signrank(complexity(highcomplexity_group_subjidx,1), complexity(highcomplexity_group_subjidx,3),"Tail","left");
    WilcoxonTests.response_time_highcomplexity(1) = signrank(response_time(highcomplexity_group_subjidx,1), response_time(highcomplexity_group_subjidx,2),"Tail","left");
    WilcoxonTests.response_time_highcomplexity(2) = signrank(response_time(highcomplexity_group_subjidx,2), response_time(highcomplexity_group_subjidx,3),"Tail","left");
    WilcoxonTests.response_time_highcomplexity(3) = signrank(response_time(highcomplexity_group_subjidx,1), response_time(highcomplexity_group_subjidx,3),"Tail","left");



    LME.complexity_group_subjidx=complexity_group_subjidx;

    % Counterbalanced trial design
    if(experiment ~= "exp3")  
        rewards_phase = cell(length(cond),1);
        for c=1:length(cond)
            for phase=1:8
                for subj=1:n_subj
                    idx = find(data(subj).cond==cond(c));
                    subj_r = data(subj).r(idx);
                    reward_phase(subj,phase) = mean(subj_r(phase:8:end));
                end
            end
            rewards_phase{c} = reward_phase;
            CohensD.undesired_cycle_effects = table2cell(meanEffectSize(reward_phase(:,1), reward_phase(:,end), Effect="cohen", Paired=true));
            [~,TTests.undesired_cycle_effects,TTestsCI.undesired_cycle_effects] = ttest(reward_phase(:,1), reward_phase(:,end), "Tail","left");
            WilcoxonTests.undesired_cycle_effects = signrank(reward_phase(:,1), reward_phase(:,end), "Tail","left");
        end
    else
        rewards_phase = cell(length(set_sizes),1);
        periods = [2*4,4*2,6*2];
        for set_size_idx = 1:length(set_sizes)
            period = periods(set_size_idx);
            data = datas.(task_names(set_size_idx));
            for phase=1:period
                for subj=1:n_subj
                    subj_r = data(subj).r;
                    reward_phase(subj,phase) = mean(subj_r(phase:period:end));
                end
            end
            rewards_phase{set_size_idx} = reward_phase;
            CohensD.undesired_cycle_effects = table2cell(meanEffectSize(reward_phase(:,1), reward_phase(:,end), Effect="cohen", Paired=true));
            [~,TTests.undesired_cycle_effects(set_size_idx),TTestsCI.undesired_cycle_effects(:,set_size_idx)] = ttest(reward_phase(:,1), reward_phase(:,end), "Tail","left");
            WilcoxonTests.undesired_cycle_effects(set_size_idx) = signrank(reward_phase(:,1), reward_phase(:,end), "Tail","left");
        end
    end
    BehavioralStats.rewards_phase = rewards_phase;

    % Cohen's d--only return the point estimate
    CohensD_new = struct();
    fields = fieldnames(CohensD);
    for i = 1:numel(fields)
        fieldName = fields{i};
        % Extract the first column of the 3x2 cell and convert it to a 1x3 vector
        CohensD_new.(fieldName) = cell2mat(CohensD.(fieldName)(:, 1))';
    end
    CohensD = CohensD_new;
end

%% Figure 1 partial
function [] = Figure1(exps, figspecs, cmap, cmap_exp3)
    exp1 = exps.exp1; exp3 = exps.exp3;
    markersize = 20; linewidth=1.5;
    feedback_duration = 0.3; 
    cond = [0,500,2000];
    exp1_R = exps.exp1.optimal_sol.R;
    exp1_V = exps.exp1.optimal_sol.V;
    RT_intercept = 0.3; RT_slope = 0.45;
    exp1_RT = exp1_R .* RT_slope + RT_intercept;
    exp1_rewardrate = zeros(length(exp1_R), length(cond));
    for c=1:length(cond)
        exp1_rewardrate(:,c) = exp1_V ./ (exp1_RT + feedback_duration + cond(c)./1000);
    end
    
    exp3_R = exps.exp3.optimal_sol.R;
    exp3_V = exps.exp3.optimal_sol.V;
    set_sizes = [2,4,6];
    task_names = convertCharsToStrings(fieldnames(exps.exp3.optimal_sol.R));
    exp3_rewardrate = zeros(length(exp3_R.(task_names(1))), length(task_names));
    for task = 1:length(task_names)
        task_name = task_names(task);
        exp3_RT.(task_name) = exp3_R.(task_name) .* RT_slope + RT_intercept;
        exp3_rewardrate(:,task) = exp3_V.(task_name) ./ (exp3_RT.(task_name) + feedback_duration + 2);
    end
    
    
    % Figures
    figure("Position", figspecs)
    tiledlayout(6,2, 'Padding', 'loose', 'TileSpacing', 'compact'); 
    ttl_position_shift = -0.18;
    ttl_fontsize = 10;

    nexttile(1,[2,1]); hold on;
    plot(exp1_R, exp1_V, "k-","MarkerSize", markersize, "LineWidth", linewidth)
    xlabel("Policy complexity (bits)")
    ylabel("Trial-averaged reward")
    xticks(0:0.5:log2(4))
    xlim([0 log2(4)])
    yticks(0.3:0.2:1)
    ylim([0.3,0.8])
    ttl = title('D', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_shift-0.02; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = 1.05; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
    nexttile(5,[2,1]); hold on;
    plot(exp3_R.(task_names(end)), exp3_RT.(task_names(end)), "k-","MarkerSize", markersize, "LineWidth", linewidth)
    xlabel("Policy complexity (bits)")
    ylabel("RT (s)")
    xticks(0:0.5:log2(6))
    xlim([0 log2(6)])
    yticks(0:0.5:1.5)
    ylim([0,Inf])
    ttl = title('E', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_shift-0.02; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = 1.05; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
    nexttile(9,[2,1]); hold on;
    for task=1:length(task_names)
        plot(exp3_R.(task_names(task)), exp3_V.(task_names(task)), "-","Color", cmap_exp3(task,:), "MarkerSize", markersize, "LineWidth", linewidth)
    end
    xlabel("Policy complexity (bits)")
    ylabel("Trial-averaged reward")
    xticks(0:0.5:log2(6))
    xlim([0 log2(6)])
    yticks(0:0.25:1)
    ylim([0,1])
    h_leg = legend("Set size = 2","Set size = 4", "Set size = 6", "location","southeast");
    h_leg.BoxFace.ColorType='truecoloralpha';
    h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
    ttl = title('G', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_shift-0.02; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = 1.05; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    nexttile(2, [3,1]); hold on;
    for c=1:length(cond)
        plot(exp1_R, normalize(exp1_rewardrate(:,c),"range",[0,1]),"-","Color", cmap(c,:), "MarkerSize", markersize, "LineWidth", linewidth)
        [max_rr, max_rr_idx] = max(exp1_rewardrate(:,c));
        max_rr_complexity = exp1_R(max_rr_idx);
        plot([max_rr_complexity,max_rr_complexity],[0,1], "-","Color", [cmap(c,:),0.3], "MarkerSize", markersize, "LineWidth", linewidth,"HandleVisibility","off")
    end
    xlabel("Policy complexity (bits)")
    ylabel({" ","Time-averaged reward (/s), ","normalized"})
    xticks(0:0.5:log2(4))
    xlim([0 log2(4)])
    yticks(0:0.25:1)
    ylim([0,1])
    ttl = title('F', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_shift-0.07; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    h_leg = legend("ITI = 0s","ITI = 0.5s","ITI = 2s", "location","south");
    h_leg.BoxFace.ColorType='truecoloralpha';
    h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

    nexttile(8, [3,1]); hold on;
    for task=1:length(task_names)
        plot(exp3_R.(task_names(task)), normalize(exp3_rewardrate(:,task),"range",[0,1]),"-","Color", cmap_exp3(task,:), "MarkerSize", markersize, "LineWidth", linewidth)
        [max_rr, max_rr_idx] = max(exp3_rewardrate(:,task));
        max_rr_complexity = exp3_R.(task_names(task))(max_rr_idx);
        plot([max_rr_complexity,max_rr_complexity],[0,1], "-","Color", [cmap_exp3(task,:),0.3], "MarkerSize", markersize, "LineWidth", linewidth, "HandleVisibility","off")
    end
    xlabel("Policy complexity (bits)")
    ylabel({" ","Time-averaged reward (/s), ","normalized"})
    xticks(0:0.5:log2(6))
    xlim([0 log2(6)])
    yticks(0:0.25:1)
    ylim([0,1])
    ttl = title('H', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_shift-0.07; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    h_leg = legend("Set size = 2","Set size = 4", "Set size = 6", "location","south");
    h_leg.BoxFace.ColorType='truecoloralpha';
    h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

end

%% Figure 2 partial
function [] = Figure2(exp, figspecs)
    Q = exp.optimal_sol.Q;
    num_states=length(Q);
    Q_transpose = Q';

    figure('Position', figspecs);
    tiledlayout(1,1, 'Padding', 'compact', 'TileSpacing', 'tight'); 
    reward_mat_fontsize=12;
    ax=nexttile(1,[1,1]); hold on;
    ax.XAxis.FontSize = reward_mat_fontsize; ax.YAxis.FontSize = reward_mat_fontsize;
    [X Y]=meshgrid(1:num_states,1:num_states);
    string = mat2cell(num2str(Q_transpose(:)),ones(num_states*num_states,1));
    imagesc(zeros(num_states),'AlphaData',0)
    text(Y(:)-0.33,flipud(X(:)),string,'HorizontalAlignment','left', 'fontsize',reward_mat_fontsize+3)
    grid = .5:1:(num_states+0.5);
    grid1 = [grid;grid];
    grid2 = repmat([.5;(num_states+0.5)],1,length(grid));
    plot(grid1,grid2,'k')
    plot(grid2,grid1,'k')
    set(gca,'TickLength',[0,0])
    set(gca,'xaxisLocation','top')
    xlim([0.5, num_states+0.5])
    ylim([0.5, num_states+0.5])
    xticks(1:num_states)
    yticks(1:num_states)
    xticklabels("A_"+[1:4])
    yticklabels("S_"+[4:-1:1])
end

%% Figure 3
function [] = Figure3(experiment_stats, cmap, figspecs)
    markersize = 20; linewidth=1.5;
    cond = [0,500,2000];

    % Parse participant behavioral stats
    accuracy = experiment_stats.BehavioralStats.accuracy;
    reward = experiment_stats.BehavioralStats.reward;
    complexity = experiment_stats.BehavioralStats.complexity;
    response_time = experiment_stats.BehavioralStats.response_time;
    cond_entropy = experiment_stats.BehavioralStats.cond_entropy;
    repeat_actions = experiment_stats.BehavioralStats.repeat_actions;
    reward_rate = experiment_stats.BehavioralStats.reward_rate;
    difficulties = experiment_stats.BehavioralStats.difficulties;
    V = experiment_stats.optimal_sol.V;
    R = experiment_stats.optimal_sol.R;
    
    % Figure 1: policy complexity, reward rate, RT
    figure('Position', figspecs);
    tiledlayout(2,10, 'Padding', 'loose', 'TileSpacing', 'tight'); 
    ttl_fontsize = 10; ttl_position_xshift = -0.27; ttl_position_yshift=1.08;

    % reward vs complexity
    ax=nexttile(1,[2,4]);
    hold on; colororder(cmap)
    plot(ax,complexity,reward,'.','MarkerSize',markersize-5)
    plot(ax,R, V,'k', 'LineWidth', linewidth);
    xticks(0:0.5:2)
    xticklabels(0:0.5:2)
    yticks(0:0.25:1)
    xlim([0,2])
    ylim([0,1])
    ylabel('Trial-averaged reward')
    xlabel('Policy complexity (bits)')
    h_leg = legend('ITI = 0s','ITI = 0.5s','ITI = 2s', 'Location', "northwest");
    h_leg.BoxFace.ColorType='truecoloralpha';
    h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
    ttl = title('A', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift+0.13; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift-0.05; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
    
    % optimal complexity vs ITI
    [n_subj,~] = size(response_time);
    ax=nexttile([1,2]); hold on;
    [se,m] = wse(complexity);
    errorbar(ax,cond/1000,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    xlabel('ITI (s)')
    xticks(cond/1000)
    xticklabels(cond/1000)
    ylabel('Policy complexity (bits)')
    yticks(0.2:0.2:0.8)
    ylim([0.2 0.8])
    xlim([-0.25 2.25])
    ttl = title('B', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
    % optimal RT vs ITI
    ax=nexttile([1,2]); hold on;
    [se,m] = wse(response_time);
    errorbar(ax,cond/1000,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    xlabel('ITI (s)')
    xticks(cond/1000)
    xticklabels(cond/1000)
    ylabel('RT (s)')
    yticks(0.2:0.2:0.8)
    xlim([-0.25 2.25])
    ylim([0.2 0.8])
    ttl = title('C', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % H(A|S)
    ax=nexttile([1,2]); hold on;
    [se,m] = wse(cond_entropy);
    errorbar(ax,[0,0.5,2], m,se, "k.-", "MarkerSize",markersize,"LineWidth",linewidth)
    %xlabel("ITI (s)")
    xticks(cond/1000)
    xticklabels(cond/1000)
    xlabel("ITI (s)")
    ylabel("H(A|S)")
    yticks(0.8:0.2:1.6)
    xlim([-0.25,2.25])
    ylim([0.8,1.6])
    ttl = title('D', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
    % Perseverance
    ax=nexttile([1,2]); hold on;
    [se,m] = wse(repeat_actions);
    errorbar(ax,[0,0.5,2], m,se, "k.-", "MarkerSize",markersize,"LineWidth",linewidth)
    xticks(cond/1000)
    xticklabels(cond/1000)
    xlim([-0.25,2.25])
    ylim([0.3,0.6]) %ylim([0.2,0.6])
    yticks(0.3:0.1:0.6)
    xlabel("ITI (s)")
    ylabel("P(repeat previous action)")
    ttl = title('E', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    ax=nexttile([1,2]); hold on;
    [se,m] = wse(reward);
    errorbar(ax, cond/1000, m,se,'.-','MarkerSize',20,'LineWidth',1.5,'Color','k')
    xlabel('ITI (s)')
    ylabel('Trial-averaged reward (/s)')
    xticks(cond/1000)
    xticklabels(cond/1000)
    xlim([-0.25 2.25])
    ylim([0.4,0.55])%ylim([0.4,0.6])
    yticks(0.4:0.05:0.55)
    ttl = title('F', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    ax=nexttile([1,2]); hold on;
    [se,m] = wse(reward_rate);
    errorbar(ax, cond/1000, m,se,'.-','MarkerSize',20,'LineWidth',1.5,'Color','k')
    xlabel('ITI (s)')
    ylabel('Time-averaged reward (/s)')
    xticks(cond/1000)
    xticklabels(cond/1000)
    xlim([-0.25 2.25])
    ylim([0,1])
    yticks(0:0.2:1)
    ttl = title('G', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

end


%% Fig. 4
function [] = Figure4(experiment_stats, cmap, figspecs)
    markersize = 20; linewidth=1.5;
    cond = [0,500,2000];
    feedback_duration=0.3; % seconds
    example_subj_idx = 42;

    % Parse participant behavioral stats
    complexity = experiment_stats.BehavioralStats.complexity;
    response_time = experiment_stats.BehavioralStats.response_time;
    reward_rate = experiment_stats.BehavioralStats.reward_rate;
    complexity_rrmax = experiment_stats.LME.complexity_rrmax;
    difficulties = experiment_stats.BehavioralStats.difficulties;
    n_subj = length(response_time);
    V = experiment_stats.optimal_sol.V;
    R = experiment_stats.optimal_sol.R;
    

    response_time_flat = response_time(:);
    RT_lme = experiment_stats.LME.RT_lme;
    RT_lme_theoretical = experiment_stats.LME.RT_lme_theoretical;

    % Figure: policy complexity, reward rate, RT
    figure('Position', figspecs);
    tiledlayout(2,3, 'Padding', 'tight', 'TileSpacing', 'tight'); 
    ttl_fontsize = gca().FontSize; ttl_position_xshift = -0.21; ttl_position_yshift = 1.02;

    % Perceived difficulty
    nexttile; hold on;
    [se,m] = wse(difficulties);
    errorbar(cond./1000, m,se, "k.-", 'MarkerSize',markersize,"LineWidth",linewidth)
    xticks(cond/1000)
    xticklabels(cond/1000)
    xlim([-0.25,2.25])
    ylim([1,3])
    xlabel("ITI (s)")
    ylabel("Perceived difficulty ranking")
    ttl = title('A', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % RT_true vs. RT_pred
    nexttile; hold on;
    for c=1:length(cond)
        plot(response_time_flat(((c-1)*n_subj+1):(c*n_subj)), RT_lme(((c-1)*n_subj+1):(c*n_subj)), ".", "MarkerSize",markersize*0.6, 'Color', cmap(c,:));
    end
    max_RT = max(response_time_flat)*1.05;
    plot([0,max_RT],[0,max_RT],"k:", "LineWidth",linewidth)
    xlim([0,min(2,max_RT)])
    ylim([0,min(2,max_RT)])
    xlabel("RT (s)")
    ylabel("RT_{pred} (s)")
    h_leg = legend('ITI = 0s','ITI = 0.5s','ITI = 2s', 'Location', "northwest");
    h_leg.BoxFace.ColorType='truecoloralpha';
    h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
    ttl = title('B', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % All the linear RTH curves
    nexttile;
    plot(repmat(R', n_subj,1)', RT_lme_theoretical','-','LineWidth',linewidth,'Color', [0,0,0,0.1]);
    yticks(0:0.5:2)
    xticks(0:0.5:2)
    ylim([0,min(2,max_RT)])
    xlim([0,2])
    ylabel("RT_{pred} (s)")
    xlabel("Policy complexity (bits)")
    set(gca,'box','off')
    ttl = title('C', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % Leftward bias
    titles = ["D","E","F"];
    zero_line_height = [0.35,0.25,0.25]; 
    ymax = [0.3,0.2,0.2];
    for c=1:length(cond)
        nexttile; hold on;
        histogram(complexity(:,c)-complexity_rrmax(:,c), -2:0.05:2, "Normalization","probability", "FaceColor", cmap(c,:), 'EdgeColor','none')
        ylabel("Relative Frequency")
        plot([0,0],[0,zero_line_height(c)], ":","Color",[0,0,0,1], "LineWidth", linewidth);
        xticks(-2:1:2)
        xlim([-2,2])
        ylim([0,ymax(c)])
        xlabel("Policy complexity, optimal - empirical (bits)")
        %title("ITI="+cond(c)/1000+"s")
        ttl = title(titles(c), "Fontsize", ttl_fontsize);
        ttl.Units = 'Normalize'; 
        ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
        ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
        ttl.HorizontalAlignment = 'left'; 
        set(gca,'box','off')
    end
end

%% Figure 5
function [] = Figure5(experiment_stats, cmap, figspecs)
    markersize = 20; linewidth=1.5;
    cond = [0,500,2000];
    feedback_duration=0.3; % seconds

    % Parse participant behavioral stats
    complexity = experiment_stats.BehavioralStats.complexity;
    response_time = experiment_stats.BehavioralStats.response_time;
    reward_rate = experiment_stats.BehavioralStats.reward_rate;
    P_a_given_s = experiment_stats.BehavioralStats.P_a_given_s;
    P_a_perserv_given_suboptimal_s_mean = experiment_stats.BehavioralStats.SuboptimalA;
    complexity_rrmax = experiment_stats.LME.complexity_rrmax;
    n_subj = length(response_time);
    V = experiment_stats.optimal_sol.V;
    R = experiment_stats.optimal_sol.R;
    Q = experiment_stats.optimal_sol.Q;
    Q_transpose = Q';

    %% Plot the reward probs
    num_states=4;
    figure('Position', figspecs);
    tiledlayout(1,4, 'Padding', 'compact', 'TileSpacing', 'tight'); 
    ttl_fontsize = gca().FontSize; ttl_position_xshift = -0.3; ttl_position_yshift = 1.05;

    reward_mat_fontsize=10;
    ax=nexttile(1,[1,1]); hold on;
    ax.XAxis.FontSize = reward_mat_fontsize; ax.YAxis.FontSize = reward_mat_fontsize;
    [X Y]=meshgrid(1:num_states,1:num_states);
    string = mat2cell(num2str(Q_transpose(:)),ones(num_states*num_states,1));
    imagesc(zeros(num_states),'AlphaData',0)
    text(Y(:)-0.33,flipud(X(:)),string,'HorizontalAlignment','left', 'fontsize',reward_mat_fontsize+3)
    grid = .5:1:(num_states+0.5);
    grid1 = [grid;grid];
    grid2 = repmat([.5;(num_states+0.5)],1,length(grid));
    plot(grid1,grid2,'k')
    plot(grid2,grid1,'k')
    set(gca,'TickLength',[0,0])
    set(gca,'xaxisLocation','top')
    xlim([0.5, num_states+0.5])
    ylim([0.5, num_states+0.5])
    xticks(1:num_states)
    yticks(1:num_states)
    xticklabels("A_"+[1:4])
    yticklabels("S_"+[4:-1:1])
    ttl = title('A', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift+0.09; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % optimal complexity vs ITI
    [n_subj,~] = size(response_time);
    ax=nexttile([1,1]); hold on;
    [se,m] = wse(complexity);
    errorbar(ax,cond/1000,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    xlabel('ITI (s)')
    xticks(cond/1000)
    xticklabels(cond/1000)
    ylabel('Policy complexity (bits)')
    ylim([0 0.5])
    xlim([-0.25 2.25])
    ttl = title('B', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
    % optimal RT vs ITI
    ax=nexttile([1,1]); hold on;
    [se,m] = wse(response_time);
    errorbar(ax,cond/1000,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    xlabel('ITI (s)')
    xticks(cond/1000)
    xticklabels(cond/1000)
    ylabel('RT (s)')
    xlim([-0.25 2.25])
    ylim([0.2 0.7])
    ttl = title('C', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % P(a_perserv | s_suboptim)
    nexttile([1,1]); hold on
    [se,m] = wse(P_a_perserv_given_suboptimal_s_mean);
    errorbar(cond/1000, m,se, "k.-",'MarkerSize',20,'LineWidth',1.5);
    xlabel("ITI (s)")
    xticks(cond/1000)
    xticklabels(cond/1000)
    ylabel("P(a_1 | s_3 or s_4)")
    ylim([0.3,0.45])
    xlim([-0.25,2.25])
    ttl = title('D', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
end

%% Figure 6
function [] = Figure6(experiment_stats, cmap, figspecs)
    markersize = 20; linewidth=1.5;
    cond = [2,4,6];
    num_states = max(cond);

    % Parse participant behavioral stats
    accuracy = experiment_stats.BehavioralStats.accuracy;
    reward = experiment_stats.BehavioralStats.reward;
    complexity = experiment_stats.BehavioralStats.complexity;
    response_time = experiment_stats.BehavioralStats.response_time;
    cond_entropy = experiment_stats.BehavioralStats.cond_entropy;
    repeat_actions = experiment_stats.BehavioralStats.repeat_actions;
    reward_rate = experiment_stats.BehavioralStats.reward_rate;
    difficulties = experiment_stats.BehavioralStats.difficulties;
    complexity_rrmax = experiment_stats.LME.complexity_rrmax;
    V = experiment_stats.optimal_sol.V;
    R = experiment_stats.optimal_sol.R;

    complexity_lba = experiment_stats.LBA.complexity_lba;
    response_time_lba = experiment_stats.LBA.response_time_lba;
    repeat_actions_lba = experiment_stats.LBA.repeat_actions_lba;
    lba_models = size(complexity_lba);
    lba_models = lba_models(1);

    % Figure 1: policy complexity, reward rate, RT
    figure('Position', figspecs);
    tiledlayout(2,10, 'Padding', 'loose', 'TileSpacing', 'compact'); 
    ttl_fontsize = 10; ttl_position_xshift = -0.27; ttl_position_yshift=1.08;

    % reward vs complexity
    ax=nexttile(1,[2,4]);
    hold on;
    for c=1:length(cond)
        plot(ax,complexity(:,c),reward(:,c),'.','MarkerSize',markersize-5, "Color", cmap(c,:))
        plot(ax,R.("set_size_"+cond(c)),V.("set_size_"+cond(c)),'-','LineWidth', linewidth, "Color", cmap(c,:), "HandleVisibility","off")
    end
    xticks(0:0.5:log2(num_states))
    xticklabels(0:0.5:log2(num_states))
    yticks(0:0.25:1)
    ylim([0,1])
    xlim([0,log2(num_states)])
    ylabel('Trial-averaged reward')
    xlabel('Policy complexity (bits)')
    h_leg = legend('Set size = 2','Set size = 4','Set size = 6', 'Location', "northwest");
    h_leg.BoxFace.ColorType='truecoloralpha';
    h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
    ttl = title('A', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift+0.13; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift-0.05; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
    % optimal complexity vs ITI
    [n_subj,~] = size(response_time);
    ax=nexttile([1,2]); hold on;
    [se,m] = wse(complexity);
    errorbar(ax,cond,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    %xlabel('ITI (s)')
    xticks(cond)
    xticklabels(cond)
    xlabel("Set size")
    ylabel('Policy complexity (bits)')
    yticks(0:0.2:1)
    ylim([0 1.1])
    xlim([1.5 6.5])
    for m=1:lba_models
        complexity_lbamodel = squeeze(complexity_lba(m,:,:));
        [se_lba,m_lba] = wse(complexity_lbamodel);
        errorbar(cond,m_lba,se_lba, ".:","MarkerSize",20,"LineWidth",1, 'Color', cmap(m+length(cond),:))
    end
    %h_leg=legend('Human','LBA', 'Location', "southeast");
    h_leg=legend('Human','LBA 1','LBA 2', 'LBA 3', 'Location', "northwest");
    h_leg.BoxFace.ColorType='truecoloralpha';
    h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
    ttl = title('B', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    
    % optimal RT vs ITI
    ax=nexttile([1,2]); hold on;
    [se,m] = wse(response_time);
    errorbar(ax,cond,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    %xlabel('ITI (s)')
    xticks(cond)
    xticklabels(cond)
    xlabel("Set size")
    ylabel('RT (s)')
    yticks(0.2:0.2:0.8)
    xlim([1.5 6.5])
    ylim([0.2 0.8])
    for m=1:lba_models
        response_time_lbamodel = squeeze(response_time_lba(m,:,:))./1000;
        [se_lba,m_lba] = wse(response_time_lbamodel);
        errorbar(cond,m_lba, se_lba, ".:","MarkerSize",20,"LineWidth",1, 'Color', cmap(m+length(cond),:))
    end
    ttl = title('C', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    
    % Perseverance
    ax=nexttile([1,2]); hold on;
    [se,m] = wse(repeat_actions);
    errorbar(ax,cond, m,se, "k.-", "MarkerSize",markersize,"LineWidth",linewidth)
    xticks(cond)
    xticklabels(cond)
    xlim([1.5,6.5])
    ylim([0.1,0.5]) %ylim([0.2,0.6])
    yticks(0.1:0.1:0.5)
    xlabel("Set size")
    ylabel("P(repeat previous action)")
    for m=1:lba_models
        repeat_actions_lbamodel = squeeze(repeat_actions_lba(m,:,:));
        [se_lba,m_lba] = wse(repeat_actions_lbamodel);
        errorbar(cond,m_lba, se_lba, ".:","MarkerSize",20,"LineWidth",1, 'Color', cmap(m+length(cond),:))
    end
    ttl = title('D', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % Difficulty
    nexttile([1,2]); hold on;
    [se,m] = wse(difficulties);
    errorbar(cond, m,se, "k.-", 'MarkerSize',markersize,"LineWidth",linewidth)
    xticks(cond)
    xticklabels(cond)
    xlim([1.5,6.5])
    ylim([1,3])
    xlabel("Set size")
    ylabel("Perceived difficulty ranking")
    ttl = title('E', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % Leftwatrd bias
    ax=nexttile([1,4]); hold on;
    bin_min=-2.5;
    bin_max = 1;
    bin_width = 0.1;
    for c=1:length(cond)
        histogram(complexity(:,c)-complexity_rrmax(:,c), bin_min:bin_width:bin_max, "Normalization","probability", "FaceColor", cmap(c,:), 'EdgeColor','none')
        xticks(bin_min:0.5:bin_max)
        xlim([bin_min,bin_max])
    end
    plot([0,0],[0,0.15], ":","Color",[0,0,0,1], "LineWidth", linewidth);
    ylim([0 0.15])
    ylabel("Relative Frequency")
    xlabel("Policy complexity, optimal - empirical (bits)")
    set(gca,'box','off')
    h_leg = legend('Set size = 2','Set size = 4','Set size = 6', 'Location', "northeast");
    h_leg.BoxFace.ColorType='truecoloralpha';
    h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
    ttl = title('F', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift./2; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 


end


%% Figure S1
function [] = FigureS1(exps, experiment_names, figspecs, cmap, cmap_exp3)
    markersize = 20; linewidth=1.5;
    figure("Position", figspecs)
    tiledlayout(3,1, 'Padding', 'tight', 'TileSpacing', 'tight'); 
    ttl_fontsize = gca().FontSize; ttl_position_xshift = -0.2; ttl_position_yshift = 1.1;

    titles = ["A","B","C"];
    for exp=1:length(experiment_names)
        nexttile; hold on
        experiment = experiment_names(exp);
        rewards_phase = exps.(experiment).BehavioralStats.rewards_phase;
        for c=1:3
            reward_phase = squeeze(rewards_phase{c});
            dims = size(reward_phase);
            n_subj = dims(1);
            period = dims(2);
            [se,m] = wse(reward_phase);
            if(experiment~="exp3")
                errorbar((1:period)',m,se, ".-","Color",cmap(c,:), "MarkerSize", markersize, "LineWidth", linewidth);
            else
                errorbar((1:period)',m,se, ".-","Color",cmap_exp3(c,:), "MarkerSize", markersize, "LineWidth", linewidth);
            end
        end
        if(experiment=="exp1")
            ylim([0.4,0.6])
            h_leg = legend("ITI = 0s", "ITI = 0.5s", "ITI = 2s", "location", "northwest");
            h_leg.BoxFace.ColorType='truecoloralpha';
            h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
        elseif (experiment=="exp3")
            ylim([0.3,0.5])
            h_leg = legend("Set size = 2", "Set size = 4", "Set size = 6");
            h_leg.BoxFace.ColorType='truecoloralpha';
            h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
        else
            ylim([0.4,0.5])
        end
        ylabel({"\bfExperiment "+exp+"\rm", "Trial-averaged reward"})
        %ylabel("Trial-averaged reward")
        xlim([0.5,period+0.5])

        % ttl = title(titles(exp), "Fontsize", ttl_fontsize);
        % ttl.Units = 'Normalize'; 
        % ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
        % ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
        % ttl.HorizontalAlignment = 'left'; 
    end
    % xlabel("Trial phase in counterbalanced cycle")
    xlabel("Trial index in counterbalanced run")
end


%% Figure S2: subgroup analysis
function [] = FigureS2(exps, experiment_names, figspecs)
    markersize=20; linewidth=1.5;
    figure("Position", figspecs)
    tiledlayout(3,4, 'Padding', 'loose', 'TileSpacing', 'tight');
    group_titles = ["Low complexity group", "High complexity group"];
    for exp=1:length(experiment_names)
        experiment = experiment_names(exp);
        if(experiment~="exp3")
            cond=[0,0.5,2];
            xlims = [-0.25, 2.25];
            xlab = "ITI (s)";
        else
            cond=[2,4,6];
            xlims = [1.5,6.5];
            xlab = "Set size"
        end
        complexity = exps.(experiment).BehavioralStats.complexity;
        response_time = exps.(experiment).BehavioralStats.response_time;
        group_subjidx = exps.(experiment).LME.complexity_group_subjidx;
        
        for group=1:2
            nexttile; hold on
            [se,m] = wse(complexity(group_subjidx{group},:));
            errorbar(cond,m,se, "k.-", "MarkerSize", markersize, "LineWidth", linewidth);
            xticks(cond)
            xticklabels(cond)
            xlim([xlims])
            xlabel(xlab)
            if(group==1)
                ylabel({"\bfExperiment "+exp+"\rm", "Policy complexity (bits)"})
            else
                ylabel("Policy complexity (bits)")
            end
            if(exp==1)
                title(group_titles(group))
            end
        end
        for group=1:2
            nexttile; hold on
            [se,m] = wse(response_time(group_subjidx{group},:));
            errorbar(cond,m,se, "k.-", "MarkerSize", markersize, "LineWidth", linewidth);
            xticks(cond)
            xticklabels(cond)
            xlim([xlims])
            xlabel(xlab)
            ylabel("RT (s)")
            if(exp==1)
                title(group_titles(group))
            end
        end
    end
end

%% Figure S3: All experiment behavioral and LME results.
function [] = FigureS3(experiments_stats, cmap, cmap_exp3, figspecs)
    figure('Position', figspecs);
    tiledlayout(3,6, 'Padding', 'none', 'TileSpacing', 'tight'); 
    ttl_fontsize = 10; ttl_position_xshift = -0.3; ttl_position_yshift = 1.05;
    for exp=1:3
        experiment = "exp"+exp;
        experiment_stats = experiments_stats.(experiment);
        markersize = 20; linewidth=1.5;
        if(experiment ~= "exp3")
            cond = [0,500,2000]./1000;
        else 
            cmap = cmap_exp3;
            cond = [2,4,6];
        end
    
        reward = experiment_stats.BehavioralStats.reward;
        complexity = experiment_stats.BehavioralStats.complexity;
        response_time = experiment_stats.BehavioralStats.response_time;
        V = experiment_stats.optimal_sol.V;
        R = experiment_stats.optimal_sol.R;
        response_time_flat = response_time(:);
        RT_lme = experiment_stats.LME.RT_lme;
        RT_lme_theoretical = experiment_stats.LME.RT_lme_theoretical;
        complexity_rrmax = experiment_stats.LME.complexity_rrmax;
        n_subj = length(complexity);
    
    
        % Reward matrix Q(s,a)
        reward_mat_fontsize=8;
        ax=nexttile; hold on;
        switch experiment
            case "exp1"
                num_states = 4;
                Q = normalize(eye(num_states), 'range', [0.25 0.75]);
                text_offset = -0.25;
            case "exp2"
                num_states = 4;
                Q = normalize(eye(num_states), 'range', [0.25 0.75]);
                Q(2,:) = [0.75,0.25,0.25,0.25];
                text_offset = -0.25;
            case "exp3"
                num_states = 6;
                Q = normalize(eye(num_states), 'range', [0.25 0.75]);
                text_offset = -0.36;
        end
        Q_flat = Q';
        Q_str = num2str(Q_flat(:));
        ax.XAxis.FontSize = reward_mat_fontsize; ax.YAxis.FontSize = reward_mat_fontsize;
        [X Y]=meshgrid(1:num_states,1:num_states);
        string = mat2cell(Q_str,ones(num_states*num_states,1));
        imagesc(zeros(num_states),'AlphaData',0)
        text(Y(:)+text_offset,flipud(X(:)),string,'HorizontalAlignment','left', 'fontsize',reward_mat_fontsize)
        grid = .5:1:(num_states+0.5);
        grid1 = [grid;grid];
        grid2 = repmat([.5;(num_states+0.5)],1,length(grid));
        plot(grid1,grid2,'k')
        plot(grid2,grid1,'k')
        set(gca,'TickLength',[0,0])
        set(gca,'xaxisLocation','top')
        xlim([0.5, num_states+0.5])
        ylim([0.5, num_states+0.5])
        xticks(1:num_states)
        yticks(1:num_states)
        xticklabels("A_"+[1:num_states])
        yticklabels("S_"+[num_states:-1:1])
        if(experiment=="exp1")
            ttl = title('A', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift+0.2; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        end
        ylabel({"\bfExperiment "+exp+"\rm"}, 'FontSize', ttl_fontsize)
    
    
    
        % reward vs complexity
        ax=nexttile;
        hold on;
        for c=1:length(cond)
            plot(ax,complexity(:,c),reward(:,c),'.','MarkerSize',markersize-5, "Color", cmap(c,:))
            if(experiment=="exp3")
                plot(ax,R.("set_size_"+cond(c)),V.("set_size_"+cond(c)),'-','LineWidth', linewidth, "Color", cmap(c,:), "HandleVisibility","off")
            end
        end
        if(experiment~="exp3")
            xmax = 2;
            plot(ax,R, V,'k', 'LineWidth', linewidth);
        else 
            xmax = 2.5;
        end
        xticks(0:0.5:xmax)
        xticklabels(0:0.5:xmax)
        xtickangle(0)
        yticks(0:0.25:1)
        xlim([0,log2(num_states)])
        ylim([0,1])
        ylabel('Trial-avg. reward')
        if(experiment=="exp1")
            h_leg=legend('ITI = 0s','ITI = 0.5s','ITI = 2s', 'Location', "southeast");
            h_leg.BoxFace.ColorType='truecoloralpha';
            h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

            ttl = title('B', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        elseif(experiment=="exp3")
            xlabel('Policy complexity (bits)')
            h_leg=legend('Set size = 2','Set size = 4','Set size = 6', 'Location', "southeast");
            h_leg.BoxFace.ColorType='truecoloralpha';
            h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
        end
        if(experiment=="exp3")
            xlabel('Policy complexity (bits)')
        end
    
    
    
        % RT vs. complexity
        nexttile; hold on;
        complexity_flat = complexity(:); 
        response_time_flat = response_time(:);
        [complexity_sorted, complexity_sorted_idx] = sort(complexity_flat);
        response_time_sorted = response_time_flat(complexity_sorted_idx);
        for c=1:length(cond)
            plot(complexity(:,c), response_time(:,c),".", 'MarkerSize',markersize-5, "Color", cmap(c,:), 'HandleVisibility','off')
        end
        % mov_window_avg_RT = smoothdata(response_time_sorted,"gaussian",100);
        % plot(complexity_sorted, mov_window_avg_RT, "m:", 'LineWidth',linewidth)
        if(experiment=="exp3")
            xlabel("Policy complexity (bits)")
        end
        ylabel("Response time (s)")
        if(experiment=="exp1")
            % h_leg = legend("MWA", 'Location', "southeast");
            % h_leg.BoxFace.ColorType='truecoloralpha';
            % h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
            ttl = title('C', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        end
    
    
        % LME predicted RT vs. human actual RT
        nexttile; hold on;
        for c=1:length(cond)
            plot(response_time_flat(((c-1)*n_subj+1):(c*n_subj)), RT_lme(((c-1)*n_subj+1):(c*n_subj)), ".", "MarkerSize",markersize*0.6, 'Color', cmap(c,:));
        end
        max_RT = max(response_time_flat)*1.05;
        plot([0,max(2,max_RT)],[0,max(2,max_RT)],"k:", "LineWidth",linewidth)
        xlim([0,max(2,max_RT)])
        ylim([0,max(2,max_RT)])
        if(experiment=="exp3")
            xlabel("RT (s)")
        elseif(experiment=="exp1")
            ttl = title('D', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        end
        ylabel("RT_{pred} (s)")
    
    
        % All the fitted linear RTH curves
        nexttile;
        if(experiment~="exp3")
            plot(repmat(R', n_subj,1)', RT_lme_theoretical','-','LineWidth',linewidth,'Color', [0,0,0,0.1]);
        else
            plot(repmat(R.("set_size_6")', n_subj,1)', RT_lme_theoretical','-','LineWidth',linewidth,'Color', [0,0,0,0.1]);
        end
        yticks(0:0.5:2)
        xticks(0:0.5:2)
        ylim([0,min(2,max_RT)])
        xlim([0,2])
        ylabel("RT_{pred} (s)")
        set(gca,'box','off')
        if(experiment=="exp1")
            ttl = title('E', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        elseif(experiment=="exp3")
            xlabel("Policy complexity (bits)")
        end
    
        % Leftward bias
        nexttile; hold on
        switch experiment
            case "exp1"
                bin_min=-2; 
                bin_max = 2;
                bin_width = 0.05;
                ymax = 0.3;
            case "exp2"
                bin_min=-1;
                bin_max = 1.5;
                bin_width = 0.05;
                ymax = 0.7;
            case "exp3"
                bin_min=-3;
                bin_max = 1;
                bin_width = 0.1;
                ymax = 0.15;
        end
        for c=1:length(cond)
            histogram(complexity(:,c)-complexity_rrmax(:,c), bin_min:bin_width:bin_max, "Normalization","probability", "FaceColor", cmap(c,:), 'EdgeColor','none')
            xticks(bin_min:1:bin_max)
            xlim([bin_min,bin_max])
            plot([0,0],[0,ymax], ":","Color",[0,0,0,1], "LineWidth", linewidth);
            ylim([0,ymax])
        end
        if(experiment=="exp3")
            xlabel("I(S;A), opt-emp (bits)")
        elseif(experiment=="exp1")
            ttl = title('F', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        end
        ylabel("Relative Frequency")
        set(gca,'box','off')
    end


end


%% Figure S4: LBA results for Experiments 1 and 3
function [] = FigureS4(experiments, experiments_stats, cmap, cmap_exp3, figspecs)
    markersize = 20; linewidth=1.5;
    figure('Position', figspecs);
    tiledlayout(2,3, 'Padding', 'loose', 'TileSpacing', 'compact'); 
    ttl_fontsize = 10; ttl_position_xshift = -0.24; ttl_position_yshift=1.05;

    for exp = experiments
        experiment = "exp"+exp;
        experiment_stats = experiments_stats.(experiment);
        if(experiment=="exp3")
            cond = [2,4,6];
            cmap = cmap_exp3;
            xlims = [1.5,6.5];
            xlab = "Set size";
        else
            cond = [0,500,2000]./1000;
            xlims = [-0.25, 2.25];
            xlab = "ITI (s)";
        end
    
        % Parse participant behavioral stats
        accuracy = experiment_stats.BehavioralStats.accuracy;
        reward = experiment_stats.BehavioralStats.reward;
        complexity = experiment_stats.BehavioralStats.complexity;
        response_time = experiment_stats.BehavioralStats.response_time;
        cond_entropy = experiment_stats.BehavioralStats.cond_entropy;
        repeat_actions = experiment_stats.BehavioralStats.repeat_actions;
        reward_rate = experiment_stats.BehavioralStats.reward_rate;
        difficulties = experiment_stats.BehavioralStats.difficulties;
        complexity_rrmax = experiment_stats.LME.complexity_rrmax;
        V = experiment_stats.optimal_sol.V;
        R = experiment_stats.optimal_sol.R;
    
        complexity_lba = experiment_stats.LBA.complexity_lba;
        response_time_lba = experiment_stats.LBA.response_time_lba;
        repeat_actions_lba = experiment_stats.LBA.repeat_actions_lba;
        lba_models = size(complexity_lba);
        lba_models = lba_models(1);

        % optimal complexity vs ITI
        [n_subj,~] = size(response_time);
        ax=nexttile; hold on;
        [se,m] = wse(complexity);
        errorbar(ax,cond,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
        %xlabel('ITI (s)')
        xticks(cond)
        xticklabels(cond)
        xlabel(xlab)
        ylabel({"\bfExperiment "+exp+"\rm", "Policy complexity (bits)"})
        yticks(0:0.2:1)
        ylim([0 1.1])
        xlim(xlims)
        for m=1:lba_models
            complexity_lbamodel = squeeze(complexity_lba(m,:,:));
            [se_lba,m_lba] = wse(complexity_lbamodel);
            errorbar(cond,m_lba,se_lba, ".:","MarkerSize",20,"LineWidth",1, 'Color', cmap(m+length(cond),:))
        end
        %h_leg=legend('Human','LBA', 'Location', "southeast");
        h_leg=legend('Human','LBA 1','LBA 2', 'LBA 3', 'Location', "northwest");
        h_leg.BoxFace.ColorType='truecoloralpha';
        h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

        if(experiment=="exp1")
            ttl = title('A', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        end
        
        % optimal RT vs ITI
        ax=nexttile; hold on;
        [se,m] = wse(response_time);
        errorbar(ax,cond,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
        xticks(cond)
        xticklabels(cond)
        xlabel(xlab)
        ylabel('RT (s)')
        yticks(0.2:0.2:0.8)
        xlim(xlims)
        ylim([0.2 0.8])
        for m=1:lba_models
            response_time_lbamodel = squeeze(response_time_lba(m,:,:))./1000;
            [se_lba,m_lba] = wse(response_time_lbamodel);
            errorbar(cond,m_lba, se_lba, ".:","MarkerSize",20,"LineWidth",1, 'Color', cmap(m+length(cond),:))
        end
        if(experiment=="exp1")
            ttl = title('B', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        end
    
        
        % Perseverance
        ax=nexttile; hold on;
        [se,m] = wse(repeat_actions);
        errorbar(ax,cond, m,se, "k.-", "MarkerSize",markersize,"LineWidth",linewidth)
        xticks(cond)
        xticklabels(cond)
        xlim(xlims)
        ylim([0.1,0.5]) %ylim([0.2,0.6])
        yticks(0.1:0.1:0.5)
        xlabel(xlab)
        ylabel("P(repeat previous action)")
        for m=1:lba_models
            repeat_actions_lbamodel = squeeze(repeat_actions_lba(m,:,:));
            [se_lba,m_lba] = wse(repeat_actions_lbamodel);
            errorbar(cond,m_lba, se_lba, ".:","MarkerSize",20,"LineWidth",1, 'Color', cmap(m+length(cond),:))
        end
        if(experiment=="exp1")
            ttl = title('C', "Fontsize", ttl_fontsize);
            ttl.Units = 'Normalize'; 
            ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
            ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
            ttl.HorizontalAlignment = 'left'; 
        end
    end
end

%% Figure S5: 5 set-size conditions
function [mturkIDs, optimal_sol, BehavioralStats, LME, TTests, WilcoxonTests] = FigureS5(experiment, datas, survey, cmap, figspecs)
    feedback_duration=0.3;
    markersize = 20; linewidth=1.5;
    set_sizes = [2:6];
    cond = [2000];
    n_tasks = length(set_sizes);
    task_names = "set_size_"+set_sizes;
    n_subj = length(datas.(task_names(1)));
    
    mturkIDs = [];
    for task = 1:n_tasks
        task_name = task_names(task);
        data = datas.(task_name);
        for s = 1:n_subj
            if(task==1)
                mturkIDs = [mturkIDs; convertCharsToStrings(data(s).ID)];
            end
            iti_2_idx = data(s).cond == cond(end);
            for c = 1:length(cond)
                idx = data(s).cond == cond(c);
                state = data(s).s(idx);
                action = data(s).a(idx);
                acc = data(s).acc(idx);
                r = data(s).r(idx);
                rt = data(s).rt(idx);
                tt = data(s).tt(idx); % total time of the block
    
                % Dimensions are: [subject, set size].
                n_trials(s,task) = length(state);
                accuracy(s,task) = nanmean(acc);
                reward(s,task) = nanmean(r);
                reward_count(s,task) = sum(r);
                reward_rate(s,task) = reward_count(s,task)/tt(end);
                complexity(s,task) = mutual_information(round(state),round(action),0.1)./log(2); 
                response_time(s,task) = nanmean(rt./1000); % RT in seconds
                response_time_trialsem(s,task) = nanstd(rt./1000) ./ sqrt(n_trials(s,task));
                cond_entropy(s,task) = condEntropy(round(action), round(state));
                repeat_actions(s,task) = nanmean(action(1:end-1) == action(2:end));
            end
        end
    end
    BehavioralStats.n_trials = n_trials;
    BehavioralStats.accuracy=accuracy;
    BehavioralStats.reward=reward;
    BehavioralStats.reward_rate=reward_rate;
    BehavioralStats.complexity=complexity;
    BehavioralStats.response_time=response_time;
    BehavioralStats.cond_entropy=cond_entropy;    
    BehavioralStats.repeat_actions=repeat_actions;

    % Perceived difficulty
    T0 = table(mturkIDs,complexity,'VariableNames',["ID","policy_complexity"]);
    T1 = struct2table(survey); 
    T1 = table(table2array(T1(:,1)),table2array(T1(:,4)),'VariableNames',["ID","difficulty"]);
    T = innerjoin(T0,T1);
    if(experiment=="exp1")
        difficulties = cell2mat(table2array(T(:,3)));
    else
        difficulties = table2array(T(:,3));
    end
    BehavioralStats.difficulties=difficulties;
    
    for task =1:(n_tasks-1)
        [~,TTests.complexity(task),~,~] = ttest(complexity(:,task), complexity(:,task+1), "Tail","left");
        [~,TTests.response_time(task),~,~] = ttest(response_time(:,task), response_time(:,task+1), "Tail","left");
        [~,TTests.reward_rate(task),~,~] = ttest(reward_rate(:,task), reward_rate(:,task+1), "Tail","left");
        [~,TTests.cond_entropy(task),~,~] = ttest(cond_entropy(:,task), cond_entropy(:,task+1), "Tail","left");
        [~,TTests.repeat_actions(task),~,~] = ttest(repeat_actions(:,task), repeat_actions(:,task+1), "Tail","left");
        [~,TTests.difficulty(task),~,~] = ttest(difficulties(:,task), difficulties(:,task+1), "Tail","left");
        
        WilcoxonTests.complexity(task) = signrank(complexity(:,task), complexity(:,task+1), "Tail","left");
        WilcoxonTests.response_time(task) = signrank(response_time(:,task), response_time(:,task+1), "Tail","left");
        WilcoxonTests.reward_rate(task) = signrank(reward_rate(:,task), reward_rate(:,task+1), "Tail","left");
        WilcoxonTests.cond_entropy(task) = signrank(cond_entropy(:,task), cond_entropy(:,task+1), "Tail","left");
        WilcoxonTests.repeat_actions(task) = signrank(repeat_actions(:,task), repeat_actions(:,task+1), "Tail","left");
        WilcoxonTests.difficulty(task) = signrank(difficulties(:,task), difficulties(:,task+1), "Tail","left");
    end

    % Theoretical curves assuming linear RTH
    n_tot = 50;
    beta_set = linspace(0.1,15,n_tot);
    P_a_given_s = nan(n_subj,max(set_sizes),length(cond),max(set_sizes)); % subj, states, conds, actions
    Q_full = normalize(eye(max(set_sizes)), 'range', [0.25 0.75]);
    for set_size_idx = 1:length(set_sizes)
        set_size = set_sizes(set_size_idx);
        task_name = task_names(set_size_idx);
        p_state = ones(1,set_size)./set_size;
        Q = Q_full(1:set_size,:);
        % initialize variables
        [R.(task_name),V.(task_name),Pa.(task_name), optimal_policy.(task_name)] = blahut_arimoto(p_state,Q,beta_set);
        % P(A|S) for Experiment 3
        data = datas.(task_name);
        for subj=1:n_subj
            for state=1:set_size
                s_idx = find(data(subj).s==state);
                actions = data(subj).a(s_idx);
                [N,~] = histcounts(actions,0.5:1:(max(set_sizes)+.5));
                P_a_given_s(subj, state, set_size_idx, :) = N./sum(N);
            end
        end
    end
    optimal_sol.R = R; optimal_sol.V = V; optimal_sol.Pa = Pa; optimal_sol.optimal_policy = optimal_policy;
    BehavioralStats.P_a_given_s = P_a_given_s;
    

    % LME
    cond = set_sizes;
    complexity_flat = complexity(:); 
    response_time_flat = response_time(:);
    [complexity_sorted, complexity_sorted_idx] = sort(complexity_flat);
    response_time_sorted = response_time_flat(complexity_sorted_idx);
    LME.complexity_sorted = complexity_sorted;
    LME.response_time_sorted = response_time_sorted;
    LME.complexity_sorted_idx = complexity_sorted_idx;
    subject_id = repmat(1:n_subj, 1, length(set_sizes))';
    tbl = table(subject_id,complexity_flat,response_time_flat,'VariableNames',{'Subject','PolicyComplexity','RT'});
    lme = fitlme(tbl,'RT ~ PolicyComplexity + (1|Subject) + (PolicyComplexity-1|Subject)');
    LME.lme = lme;
    LME.tbl = tbl;
    RT_lme = predict(lme); % Return 1SD, instead of 95% CIs. 
    RT_lme_sorted = RT_lme(complexity_sorted_idx);
    LME.RT_lme = RT_lme;
    LME.RT_lme_sorted = RT_lme_sorted;

    % Counterfactual RT at different policy complexity levels
    complexity_rrmax = zeros(n_subj, length(set_sizes));
    for subj=1:n_subj
        for set_size_idx=1:length(set_sizes)
            set_size = set_sizes(set_size_idx);
            task_name = task_names(set_size_idx);
            tbl_new = table(repmat(subj,1,length(R.(task_name)))',R.(task_name),'VariableNames',{'Subject','PolicyComplexity'});
            RT_lme_theoretical = predict(lme, tbl_new);
    
            rr = V.(task_name) ./ (RT_lme_theoretical + 2000/1000 + feedback_duration);
            [max_rr, max_rr_complexity] = max(rr);
            complexity_rrmax(subj,set_size_idx) = R.(task_name)(max_rr_complexity);
        end
        RT_lme_theoreticals(subj,:) = RT_lme_theoretical; % Use R.("set_size_6") becuase its range is greatest.
    end
    LME.RT_lme_theoretical = RT_lme_theoreticals;
    LME.complexity_rrmax = complexity_rrmax;
    rhos_subj = zeros(n_subj,1);
    for subj=1:n_subj
        rhos_subj(subj) = corr(complexity(subj,:)', complexity_rrmax(subj,:)',"Type", "Spearman");
    end
    BehavioralStats.complexity_difficulty_spearman = rhos_subj;
    [~,TTests.complexity_difficulty_spearman_ispositive,~,~] = ttest(rhos_subj);
    WilcoxonTests.complexity_difficulty_spearman_ispositive = signrank(rhos_subj);

    % Leftward complexity bias
    complexity_diff_from_rrmax = complexity-complexity_rrmax;
    LME.complexity_diff_from_rrmax=complexity_diff_from_rrmax;
    for task=1:length(task_names)
        [~,TTests.complexity_lessthan_rrmax(task)] = ttest(complexity(:,task), complexity_rrmax(:,task), "Tail","left");
        WilcoxonTests.complexity_lessthan_rrmax(task) = signrank(complexity(:,task), complexity_rrmax(:,task),"Tail","left");
    end

    % Counterbalanced trial design
    reward_phase = zeros(n_subj, 8);
    for set_size_idx = 1:length(set_sizes)
        data = datas.(task_names(set_size_idx));
        for phase=1:8
            for subj=1:n_subj
                subj_r = data(subj).r;
                reward_phase(subj,phase) = mean(subj_r(phase:8:end));
                rewards_phase(subj,set_size_idx,phase) = reward_phase(subj,phase);
            end
        end
        [~,TTests.undesired_cycle_effects(set_size_idx)] = ttest(reward_phase(:,1), reward_phase(:,8), "Tail","left");
        WilcoxonTests.undesired_cycle_effects(set_size_idx) = signrank(reward_phase(:,1), reward_phase(:,8), "Tail","left");
    end
    BehavioralStats.rewards_phase = rewards_phase;
    
    % Figure
    figure('Position', figspecs);
    tiledlayout(2,3, 'Padding', 'compact', 'TileSpacing', 'tight'); 
    ttl_fontsize = gca().FontSize; ttl_position_xshift = -0.25; ttl_position_yshift = 1.02;

    % optimal complexity vs ITI
    [n_subj,~] = size(response_time);
    nexttile; hold on;
    [se,m] = wse(complexity);
    errorbar(cond,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    xticks(cond)
    xticklabels(cond)
    xlabel("Set size")
    ylabel('Policy complexity (bits)')
    yticks(0:0.2:1.2)
    ylim([0.2 1.2])
    xlim([1.5 6.5])
    ttl = title('A', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % optimal RT vs ITI
    nexttile; hold on;
    [se,m] = wse(response_time);
    errorbar(cond,m,se,'.-','MarkerSize',markersize,'LineWidth',linewidth,'Color','k')
    xticks(cond)
    xticklabels(cond)
    xlabel("Set size")
    ylabel('RT (s)')
    yticks(0.5:0.1:1)
    xlim([1.5 6.5])
    ylim([0.5 1])
    ttl = title('B', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 

    % RT_LME vs RT_true
    nexttile; hold on;
    for c=1:length(set_sizes)
        plot(response_time_flat(((c-1)*n_subj+1):(c*n_subj)), RT_lme(((c-1)*n_subj+1):(c*n_subj)), ".", "MarkerSize",markersize*0.6, 'Color', cmap(c,:));
    end
    max_RT = max(response_time_flat)*1.05;
    plot([0,max_RT],[0,max_RT],"k:", "LineWidth",linewidth)
    xlim([0,min(2.3,max_RT)])
    ylim([0,min(2.3,max_RT)])
    xlabel("RT (s)")
    ylabel("RT_{pred} (s)")
    ttl = title('C', "Fontsize", ttl_fontsize);
    ttl.Units = 'Normalize'; 
    ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
    ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
    ttl.HorizontalAlignment = 'left'; 
    % h_leg = legend("Set size = 2","Set size = 3","Set size = 4","Set size = 5","Set size = 6", 'Location', "northwest");
    % h_leg.BoxFace.ColorType='truecoloralpha';
    % h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');

    % Individual subject relationships
    exemplary_subjs = [13,22,21];
    titles = ["D","E","F"];
    for s=1:length(exemplary_subjs)
        subj = exemplary_subjs(s);
        nexttile; hold on;
        for task=1:length(task_names)
            errorbar(complexity(subj,task), response_time(subj,task), response_time_trialsem(subj,task), ".", "Color", cmap(task,:), "MarkerSize", markersize, "LineWidth",linewidth)
        end
        plot(R.(task_names(end)), RT_lme_theoreticals(subj,:), "k-", "LineWidth", linewidth)
        xlim([0,log2(6)])
        ylim([0,max(2,max(RT_lme_theoreticals(subj,:)))])
        ylabel("RT (s)")
        xlabel("Policy complexity (bits)")

        if(s==1)
            h_leg = legend("Set size = 2","Set size = 3","Set size = 4","Set size = 5","Set size = 6", 'Location', "northwest");
            h_leg.BoxFace.ColorType='truecoloralpha';
            h_leg.BoxFace.ColorData=uint8(255*[1 1 1 0.75]');
        end
        ttl = title(titles(s), "Fontsize", ttl_fontsize);
        ttl.Units = 'Normalize'; 
        ttl.Position(1) = ttl_position_xshift; % use negative values (ie, -0.1) to move further left
        ttl.Position(2) = ttl_position_yshift; % use negative values (ie, -0.1) to move further left
        ttl.HorizontalAlignment = 'left'; 
    end

end

%% Helper functions for generating data from given set of parameters. 

function [experiment_stats] = append_lba_preds(exp, experiment_stats, lba_folder)
    if(exp=="exp3")
        cond = [2,4,6];
    else
        cond = [0,500,2000];
        Nactions = 4;
    end

    n_subj = length(experiment_stats.BehavioralStats.complexity);
    % LBA results
    rng(0);
    human_data = load(lba_folder+"data/lba_data_"+exp+"_struct").datas;
    load(lba_folder+"fits/lba_"+exp+"_full", 'Params_best','model')
    sv=0.1; 
    models = 3; %3
    Datas_fitted = cell(models, 1);
    for m=1:models
        if(exp=="exp3")
            Datas_fitted{m} = fake_data_gen(exp, human_data, Params_best, model, cond, max(cond), sv, m);
        else
            Datas_fitted{m} = fake_data_gen(exp, human_data, Params_best, model, cond, Nactions, sv, m);
        end
    end
    % Find their I(S;A) and RT.
    complexity_lba = zeros(models, n_subj, length(cond));
    response_time_lba = zeros(models, n_subj, length(cond));
    repeat_actions_lba = zeros(models, n_subj, length(cond));
    for m=1:models
        for subj=1:n_subj
            data_subj = Datas_fitted{m}{subj};
            for c=1:length(cond) 
                cs = data_subj.cond;
                relevant_trials_subj_setsize = find(cs==c);
                state = data_subj.stim(relevant_trials_subj_setsize);
                action = data_subj.response(relevant_trials_subj_setsize);
                rt = data_subj.rt(relevant_trials_subj_setsize);
                complexity_lba(m,subj,c) = mutual_information(state, action,0.1)./log(2);
                response_time_lba(m,subj,c) = mean(rt);
                repeat_actions_lba(m,subj,c) = nanmean(action(1:end-1) == action(2:end));
            end
        end
    end
    experiment_stats.LBA.complexity_lba = complexity_lba;
    experiment_stats.LBA.response_time_lba = response_time_lba;
    experiment_stats.LBA.repeat_actions_lba = repeat_actions_lba;
end

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