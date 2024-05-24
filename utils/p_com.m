function [n_bits, H_pi_base, H_pi_base_given_state] = p_com(pi, pi_base, p_state)

% pi: policy [n actions x m states]
% pi_base: prob(action) with state integrated out
% p_state: probability of state

% n_bits = 0;
% for s_ind = 1:length(p_state) % how many states
%     tmp_bits = 0;
%     for a_ind = 1:length(pi_base) % how many actions
%         tmp_ent = pi(a_ind, s_ind)*log2(pi(a_ind, s_ind)/pi_base(s_ind));
%         if ~isnan(tmp_ent)
%             tmp_bits = tmp_bits + tmp_ent;
%         end
%     end
%     n_bits = n_bits + p_state(s_ind)*tmp_bits;
% end

H_pi_base = -1*sum(pi_base.*log2(pi_base));
H_pi_base_given_state = -1*sum(p_state*nansum(pi.*log2(pi), 2));
n_bits = H_pi_base - H_pi_base_given_state;