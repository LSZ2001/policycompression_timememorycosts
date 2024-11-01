function pdf = LBA_n1PDF(t, A, b, v, sv)
% Generates defective PDF for responses on node #1 (ie. normalised by
% probability of this node winning race)
%
% pdf = LBA_n1PDF(t, A, b, v, sv)
%
% SF 2012

N = size(v,2);


if(size(A,2)==1)
    % This is the original LBA code: shared A across all accumulators.
    if N > 2
        for i = 2:N
            tmp(:,i-1) = LBA_tcdf(t,A,b,v(:,i),sv);
        end
        G = prod(1-tmp,2);
    else
        G = 1-LBA_tcdf(t,A,b,v(:,2),sv);
    end
    pdf = G.*LBA_tpdf(t,A,b,v(:,1),sv);


else % Perseveration applied to A -> transform one accumulator's starting point from uniform in [0,A] to perservA+[0,A].
    if N > 2
        for i = 2:N
            tmp(:,i-1) = LBA_tcdf(t,A(:,i),b,v(:,i),sv);
        end
        G = prod(1-tmp,2);
    else
        G = 1-LBA_tcdf(t,A(:,2),b,v(:,2),sv);
    end
    pdf = G.*LBA_tpdf(t,A(:,1),b,v(:,1),sv);
    
end