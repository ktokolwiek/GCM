function GCM_wrapper()
%% Wrapper for testing the GCM model in GCM_model.m
gammas = [0.5 1 2];
forget_rates = logspace(-10,-1,7);
noise_mu = 0;
noise_sigmas = [0.1 0.5];
choice_parameters = [1 2];
no_repeats = 100;

no_combinations = length(gammas)*length(forget_rates)*length(noise_sigmas)*...
    length(choice_parameters);

C={'gamma', 'forget_rate', 'noise_mu', 'noise_sigma', 'choice_parameter',...
    'll_mean', 'll_sd'}; % the headers
%% Parallel for loop

combination = 2;
matlabpool open 12 % on the love01 machine

for gamma = gammas
    for forget_rate = forget_rates
        for noise_sigma = noise_sigmas
            for choice_parameter = choice_parameters
                lls = zeros(no_repeats,1);
                parfor iter = 1:no_repeats
                    [~, lls(iter)] = GCM_model('gamma', gamma, 'forget_rate',...
                        forget_rate, 'choice_parameter', choice_parameter,...
                        'noise_mu', noise_mu, 'noise_sigma', noise_sigma,...
                        'feedType', 1);
                end
                C(combination,:) = {gamma, forget_rate, noise_mu,...
                    noise_sigma, choice_parameter, mean(lls), std(lls)};
                combination = combination + 1;
            end
        end
    end
end


%% write out the results

[nrows,~]= size(C);

filename = 'actual_feedback.csv';
fid = fopen(filename, 'w');

fprintf(fid, '%s,%s,%s,%s,%s,%s,%s\n', C{1,:});
for row=2:nrows
    fprintf(fid, '%.1f,%.7f,%.1f,%.1f,%.1f,%.7f,%.7f\n', C{row,:});
end

fclose(fid);

matlab close
end