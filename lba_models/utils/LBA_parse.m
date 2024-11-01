function [v A b sv t0, perserv, perservA,v_lrates,p_lrates] = LBA_parse(model, pArray, Ncond)
% Parse parameter vector for fitting LBA
%
% SF 2012


%% Parse pArray vector and setup some transformations
j=1;
err = 0;
if model.v == 1
    v = repmat(pArray(j), Ncond, 1);
    j=j+1;
elseif model.v == Ncond
    for c = 1:model.v
        v(c,:) = pArray(j);
        j=j+1;
    end
else err = 1;
end

if model.A == 1
    A = repmat(pArray(j), Ncond, 1);
    j=j+1;
elseif model.A == Ncond
    for c = 1:model.A
        A(c,:) = pArray(j);
        j=j+1;
    end
else err = 1;
end

if model.b == 1
    b = repmat(pArray(j), Ncond, 1);
    j=j+1;
elseif model.b == Ncond
    for c = 1:model.b
        b(c,:) = pArray(j);
        j=j+1;
    end
else err = 1;
end

if model.sv == 1
    sv = repmat(pArray(j), Ncond, 1);
    j=j+1;
elseif model.sv == Ncond
    for c = 1:model.sv
        sv(c,:) = pArray(j);
        j=j+1;
    end
else err = 1;
end

if length(pArray) >= j
    if model.t0 == 1
        t0 = repmat(pArray(j), Ncond, 1);
        j=j+1;
    elseif model.t0 == Ncond
        for c = 1:model.t0
            t0(c,:) = pArray(j);
            j=j+1;
        end
    else err = 1;
    end
else
    t0 = repmat(200, Ncond, 1); % fix if not free param
    j=j+1;
end

% My new addition: perseveration parameters by cond
perserv = NaN;
if length(pArray) >= j
    if model.perserv == 1
        perserv = repmat(pArray(j), Ncond, 1);
        j=j+1;
    elseif model.perserv == Ncond
        for c = 1:model.perserv
            perserv(c,:) = pArray(j);
            j=j+1;
        end
    elseif ~isempty(model.perserv)
        err = 1;
    end
else
    perserv = NaN; % no perseveration if not free param
    j=j+1;
end

% My new addition: perseveration_A parameters by cond
perservA = NaN;
if length(pArray) >= j
    if model.perservA == 1
        perservA = repmat(pArray(j), Ncond, 1);
        j=j+1;
    elseif model.perservA == Ncond
        for c = 1:model.perservA
            perservA(c,:) = pArray(j);
            j=j+1;
        end
    elseif ~isempty(model.perservA) 
        err = 1;
    end
else
    perservA = NaN; % no perseveration if not free param
    j=j+1;
end

% My new addition: v_lrate parameters by cond
v_lrates = NaN;
if length(pArray) >= j
    if model.v_lrate == 1
        v_lrates = repmat(pArray(j), Ncond, 1);
        j=j+1;
    elseif model.v_lrate == Ncond
        for c = 1:model.v_lrate
            v_lrates(c,:) = pArray(j);
            j=j+1;
        end
    elseif ~isempty(model.v_lrates) 
        err = 1;
    end
else
    v_lrates = NaN; % no updating of v if not this param
    j=j+1;
end

% My new addition: v_lrate parameters by cond
p_lrates = NaN;
if length(pArray) >= j
    if model.p_lrate == 1
        p_lrates = repmat(pArray(j), Ncond, 1);
        j=j+1;
    elseif model.p_lrate == Ncond
        for c = 1:model.p_lrate
            p_lrates(c,:) = pArray(j);
            j=j+1;
        end
    elseif ~isempty(model.p_lrates) 
        err = 1;
    end
else
    p_lrates = NaN; % no updating of v if not this param
    j=j+1;
end


if err == 1
    fprintf('\n\n\nBad input! model fields can only take on values of 1 or Ncond. See help LBA_mle.\n\n');
    return;
elseif length(pArray) > j-1
    fprintf('\n\n\nBad input! pArray does not match model specification. See help LBA_mle.\n\n');
    return;
end
