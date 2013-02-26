function GCM_evaluate_model(fType, gammas, forget_rates, noise_mu, noise_sigmas, choice_parameters)
%% Wrapper for testing the GCM model in GCM_model.m
no_repeats = 1;

no_combinations = length(gammas)*length(forget_rates)*length(noise_sigmas)*...
    length(choice_parameters);
% the headers
header_train={'subj','session','feedType','trial','length_avg','length_sd','tarCat','respCat',...
    'idealCat_avg','idealCat_sd','modelledCat_avg','modelledCat_sd'};
header_test={'subj','session','feedType','trial','length_avg','length_sd','respCat',...
    'modelledCat_avg','modelledCat_sd'};

%% Parallel for loop

combination = 0;
matlabpool open 1 % on the love01 machine
fprintf('Progress is %2.0f%%',0)
for gamma = gammas
    for forget_rate = forget_rates
        for noise_sigma = noise_sigmas
            for choice_parameter = choice_parameters
                lls = zeros(no_repeats, 1);
                parfor iter = 1:no_repeats
                    [train, lls(iter), test] = GCM_model('gamma', gamma, 'forget_rate',...
                        forget_rate, 'choice_parameter', choice_parameter,...
                        'noise_mu', noise_mu, 'noise_sigma', noise_sigma,...
                        'feedType', fType, 'verbose', 100);
                    train_lengths(:,iter) = train(:,5);
                    train_ideal(:,iter) = train(:,8);
                    train_model(:,iter) = train(:,9);
                    test_lengths(:,iter) = test(:,5);
                    test_model(:,iter) = test(:,8);
                    % progress bar
                    fprintf(repmat('\b',1,length('Progress is 20p')));
                    fprintf('Progress is %2.0f%%',(combination*100*no_repeats+iter)...
                        /(no_combinations*no_repeats));
                    
                end
                combination = combination + 1;
                %% get the filenames
                if fType == 1
                    train_fname = sprintf('GCM_results/actual_training_%.1f_%.15f_%.1f_%.1f.csv',...
                        gamma, forget_rate, noise_sigma, choice_parameter);
                    test_fname = sprintf('GCM_results/actual_test_%.1f_%.15f_%.1f_%.1f.csv',...
                        gamma, forget_rate, noise_sigma, choice_parameter);
                else
                    train_fname = sprintf('GCM_results/ideal_training_%.1f_%.15f_%.1f_%.1f.csv',...
                        gamma, forget_rate, noise_sigma, choice_parameter);
                    test_fname = sprintf('GCM_results/ideal_test_%.1f_%.15f_%.1f_%.1f.csv',...
                        gamma, forget_rate, noise_sigma, choice_parameter);
                end
                %% prepare the data
                train_results = [train(:,1:4) mean(train_lengths')' std(train_lengths')' ...
                    train(:,6:7) mean(train_ideal')' std(train_ideal')' ...
                    mean(train_model')' std(train_model')'];
                test_results = [test(:,1:4) mean(test_lengths')' std(test_lengths')' ...
                    train(:,6) mean(test_model')' std(test_model')'];
                %% write out the results
                nrows=length(train(:,1));
                fid = fopen(train_fname, 'w');
                fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s\n', header_train{:});
                for row=1:nrows
                    fprintf(fid, '%d,%d,%d,%d,%.2f,%.2f,%d,%.2f,%.2f\n', test_results(row,:));
                end
                fclose(fid);
                
                nrows=length(test(:,1));
                fid = fopen(test_fname, 'w');
                fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', header_test{:});
                for row=1:nrows
                    fprintf(fid, '%d,%d,%d,%d,%.2f,%.2f,%d,%d,%.2f,%.2f,%.2f,%.2f\n', train_results(row,:));
                end
                fclose(fid);
            end
        end
    end
end


%% write out the results

[nrows,~]= size(C);

fid = fopen(filename, 'w');

fprintf(fid, '%s,%s,%s,%s,%s,%s,%s\n', C{1,:});
for row=2:nrows
    fprintf(fid, '%.1f,%.10f,%.1f,%.1f,%.1f,%.7f,%.7f\n', C{row,:});
end

fclose(fid);

matlabpool close
end
