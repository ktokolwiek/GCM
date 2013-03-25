matlabpool open
forget_rates = [0.1 0.5 0.01 0.05 0.001 0.005 0.0001 0.0005 0];
parfor for_rate = 1:length(forget_rates)
	GCM_generate_Ps_responses(forget_rates(for_rate));
end
matlabpool close
