function GCM_evaluate_model(fType, gammas, forget_rates, noise_mu, noise_sigmas, choice_parameters)
%% Wrapper for testing the GCM model in GCM_model.m
no_repeats = 100;

no_combinations = length(gammas)*length(forget_rates)*length(noise_sigmas)*...
    length(choice_parameters);
% the headers
header_train={'subj','session','feedType','trial','length_avg','length_sd',...
    'tarCat','respCat', 'idealCat','modelledCat_avg','modelledCat_sd',...
    'modelledCat_no_forget_avg','modelledCat_no_forget_sd'};
header_test={'subj','session','feedType','trial','length_avg','length_sd',...
    'respCat', 'modelledCat_avg','modelledCat_sd','modelledCat_no_forget_avg',...
    'modelledCat_no_forget_sd'};

%% Parallel for loop

combination = 0;
matlabpool open 12 % on the love01 machine
fprintf('Progress is %2.0f%%',0)
for gamma = gammas
    for forget_rate = forget_rates
        for noise_sigma = noise_sigmas
            for choice_parameter = choice_parameters
		lls = zeros(no_repeats,1);
        lls_no_forget = zeros(no_repeats,1);
                parfor iter = 1:no_repeats
                    [trainData, lls(iter), testData, lls_no_forget(iter)] =...
                        GCM_model('gamma', gamma, 'forget_rate',...
                        forget_rate, 'choice_parameter', choice_parameter,...
                        'noise_mu', noise_mu, 'noise_sigma', noise_sigma,...
                        'feedType', fType, 'verbose', -1);
                    train_lengths(:,iter) = trainData(:,5);
                    train_model(:,iter) = trainData(:,9);
                    train_model_no_forgetting(:,iter) = trainData(:,10);
                    test_lengths(:,iter) = testData(:,5);
                    test_model(:,iter) = testData(:,8);
                    test_model_no_forgetting(:,iter) = testData(:,9);
                    % progress bar
                    fprintf(repmat('\b',1,length('Progress is 20p')));
                    fprintf('Progress is %2.0f%%',(combination*no_repeats+iter)...
                        /(no_combinations*no_repeats*0.01));
                    
                end
		%% Read the files again so that we have the raw data for output like subject ID, feedback type etc.
		trainData = xlsread('../raw_data/training_all.xls');
		testData = xlsread('../raw_data/test_all.xls');
		trainData(:,8) = 2*(trainData(:,5)>30.5)-1; % -1 for cat A (short), 1 for cat B (long)
		trainData = [trainData trainData(:,6)]; % copy the feedback to modelled category.
                
		combination = combination + 1;
                %% get the filenames
                if fType == 1
                    train_fname = sprintf('../GCM_results/actual_training_%.1f_%.15f_%.1f_%.1f.csv',...
                        gamma, forget_rate, noise_sigma, choice_parameter);
                    test_fname = sprintf('../GCM_results/actual_test_%.1f_%.15f_%.1f_%.1f.csv',...
                        gamma, forget_rate, noise_sigma, choice_parameter);
                else
                    train_fname = sprintf('../GCM_results/ideal_training_%.1f_%.15f_%.1f_%.1f.csv',...
                        gamma, forget_rate, noise_sigma, choice_parameter);
                    test_fname = sprintf('../GCM_results/ideal_test_%.1f_%.15f_%.1f_%.1f.csv',...
                        gamma, forget_rate, noise_sigma, choice_parameter);
                end
                %% prepare the data
                train_results = [trainData(:,1:4) mean(train_lengths')' std(train_lengths')' ...
                    trainData(:,6:8) mean(train_model')' std(train_model')' ...
                    mean(train_model_no_forgetting')' std(train_model_no_forgetting')'];
                test_results = [testData(:,1:4) mean(test_lengths')' std(test_lengths')' ...
                    testData(:,6) mean(test_model')' std(test_model')' ...
                    mean(test_model_no_forgetting')' std(test_model_no_forgetting')'];
                %% write out the results
                fid = fopen(train_fname, 'w');
                fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', header_train{:});
                fclose(fid);
                dlmwrite(train_fname,train_results,'-append', 'precision','%.2f');
                
                fid = fopen(test_fname, 'w');
                fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,LL_%.5f,LL_sd_%.5f,NO_FORGET_LL_%.5f,LL_sd_%.5f\n',...
                    header_test{:},mean(lls),std(lls),mean(lls_no_forget),...
                    std(lls_no_forget));
                fclose(fid);
                dlmwrite(test_fname,test_results,'-append', 'precision','%.2f');
            end
        end
    end
end


matlabpool close
end
