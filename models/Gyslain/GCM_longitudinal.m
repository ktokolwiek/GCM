function GCM_longitudinal(forget_rate)
% This analyses how reestimation may lead to idealisation in a longitudinal
% study, cf. Gyslain's data from Nov 2012.

    function [training,test] = get_training_stimuli(feed_type)
        %% Distributions
        %High variance
        list_A = [6 7 8 9 10 11 12 13 14 14 15 15 16 16 17 17 18 18 19 19 ...
            20 20 21 21 21 22 22 22 23 23 23 24 24 24 25 25 25 26 26 26 27 27 27 ...
            28 28 28 29 29 29 30 30 30 31 31 31 32 32 32 33 33 34 34 35 35 36 36 ...
            37 37 38 38 39 39 40 41 42 43 44 45 46 47]';
        list_D = [14 15 16 17 18 19 20 21 22 22 23 23 24 24 25 25 26 26 27 ...
            27 28 28 29 29 29 30 30 30 31 31 31 32 32 32 33 33 33 34 34 34 35 35 ...
            35 36 36 36 37 37 37 38 38 38 39 39 39 40 40 40 41 41 42 42 43 43 44 ...
            44 45 45 46 46 47 47 48 49 50 51 52 53 54 55]';
        list_test = (1:60)';
        
        %% Add feedback
        if feed_type == 1
            % actual feedback
            list_A(:,2) = -1;
            list_D(:,2) = 1;
        elseif feed_type == 2
            % ideal feedback
            list_A(:,2) = (list_A>30.5) * 2 - 1;
            list_D(:,2) = (list_D>30.5) * 2 - 1;
        end
        
        %% Add test set
        list_test(:,2) = (list_test>30.5) * 2 - 1;
        %% Join lists
        training = [list_A;list_D];
        %% shuffle both lists
        list_test = list_test(randperm(length(list_test)),:);
        training = training(randperm(length(training)),:);
        test=list_test;
    end

    function [training,test]=read_training_stimuli(training_file, test_file, ps)
        %% Read in data
        training_raw_file = csvread(training_file);
        % category ID is in column 7, length is in column 6, feedback type in 4
        training = training_raw_file(training_raw_file(:,1)==ps,[6 7]);
        test_raw_file = csvread(test_file);
        % cat ID is 7, len is 6
        test = test_raw_file(test_raw_file(:,1)==ps,[6 7]);
    end


%%%%%%%%
%% Here we need to implement the different feedback types and how we infer
% from the length.
%% .

    function [trainingData,ll, testData] = GCM_generative_model(varargin)
        
        %% Init the model's parameters
        p=inputParser;
        addOptional(p, 'training_set', NaN);
        addOptional(p, 'test_set', NaN);
        addOptional(p, 'verbose',0,@isnumeric);
        addOptional(p, 'gamma',2,@isnumeric);
        addOptional(p, 'forget_rate',0.1,@isnumeric);
        addOptional(p, 'choice_parameter', 1, @isnumeric);
        % noise parameters, for an implementation of perceptual noise, cf.
        % general recognition theory (GRT, Ashby & Townsend, 1986)
        addOptional(p, 'noise_mu',0,@isnumeric);
        addOptional(p, 'noise_sigma',0.5, @isnumeric);
        addOptional(p, 'session',NaN);
        addOptional(p, 'feedType',NaN);
        addOptional(p, 'ps_id',NaN);
        addOptional(p, 'trial',NaN);
        parse(p,varargin{:})
        training_set = p.Results.training_set;
        test_set = p.Results.test_set;
        verbose = p.Results.verbose;
        gamma = p.Results.gamma;
        forget_rate = p.Results.forget_rate;
        choice_parameter = p.Results.choice_parameter;
        noise_mu = p.Results.noise_mu;
        noise_sigma = p.Results.noise_sigma;
        session = p.Results.session;
        feedType = p.Results.feedType;
        ps_id = p.Results.ps_id;
        trial = p.Results.trial;
        
        if verbose==100
            fprintf('Model with gamma %.1f, forget rate %.10f, choice parameter %d, noise mean %.1f, noise sd %.1f., PS: %d\n',...
                gamma, forget_rate,choice_parameter,noise_mu,noise_sigma,ps_id);
        end
        
        %% Read the training and test data
        
        trainingData = training_set;
        noInstances = length(trainingData(:,1));
        trainingData(:,3) = (trainingData(:,1)>30.5) * 2 - 1;
        % ideal feedback
        trainingData(:,1) = trainingData(:,1) + (noise_mu + noise_sigma.*randn(noInstances,1));
        % add perceptual noise
        trainingData = [trainingData trainingData(:,2)];
        % copy the feedback to modelled category
        % (1) length, (2) feedback, (3) idealCat, (4)
        % modelledCat_with_forgetting
        testData = test_set;
        testData(:,3) = (testData(:,1)>30.5) * 2 - 1;
        testData(:,1) = testData(:,1) + (noise_mu + noise_sigma.*randn(length(testData(:,1)),1));
        % add perceptual noise
        testData(:,2) = zeros(size(testData(:,1)));
        % (1) length, (2) forgetting, (3) ideal
        
        
        %% Get category memberships
        
        function [cat,ll] = get_cat_membership(inst_no)
            %% Sampling here: get up to 10 last items
            if length(presented)<10
                sample_end = length(presented)-1;
            else
                sample_end = 9;
            end
            
            presentedData=trainingData(presented(1:sample_end),:);
            len = trainingData(inst_no,1);
            oldCat = trainingData(inst_no,4); %using modelled category here,
            % it is the same as presented category to begin with anyway.
            catA = presentedData(presentedData(:,4)==-1,1);
            catB = presentedData(presentedData(:,4)==1,1);
            % just the lengths of instances in Cat A or B
            if isempty(catA) || isempty(catB)
                ll=log(0.5);
                if rand<0.5
                    cat =  -1;
                else
                    cat = 1;
                end
            else
                %% IMPLEMENT SAMPLING!!!
                %% IMPLEMENT RESAMPLING!!!
                sumCatA=sum(exp(-choice_parameter*abs(catA-len)));
                sumCatB=sum(exp(-choice_parameter*abs(catB-len)));
                % If we know the category membership of an instance then we
                % don't compare it to instances of its own category. If we
                % don't know the category membership (i.e. no feedback for
                % this instance), then we compare it to both categories.
                %if oldCat == -1
                %    sumCatA = sumCatA-1;
                %else
                %    sumCatB = sumCatB-1;
                %end % we subtract exp(0) for difference with itself.
                % We don't need to do that actually, because we are using only presented(1:end-1)
                probA=sumCatA^gamma/(sumCatA^gamma+sumCatB^gamma);
                probB=sumCatB^gamma/(sumCatA^gamma+sumCatB^gamma);
                if probA>probB
                    ll=log(probA);
                    cat = -1;
                else
                    ll=log(probB);
                    cat = 1;
                end
            end
        end
        
        %% Get the category memberships WITHOUT re-sampling
        
        function [ll] = get_ideal_membership(instances)
            lens = trainingData(:,5);
            trainingData(:,10)=zeros(length(lens),1);
            catA = trainingData(trainingData(instances,8)==-1,5);
            catB = trainingData(trainingData(instances,8)==1,5);
            sumCatA=sum(exp(-choice_parameter*pdist2(catA, lens)));
            sumCatB=sum(exp(-choice_parameter*pdist2(catB, lens)));
            probA=sumCatA.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
            probB=sumCatB.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
            indices_a = find(trainingData(:,8)==-1);
            indices_b = find(trainingData(:,8)==1);
            trainingData(probA>probB,10) = -1;
            trainingData(probB>probA,10) = 1;
            ll = sum(log(probA(indices_a)))+sum(log(probB(indices_b)));
        end
        
        %% Do one loop of forgetting
        
        function forget()
            forgotten = find(rand(1,length(presented))<forget_rate);
            presented = setdiff(presented, forgotten);
            for j=forgotten
                presented = [presented j];
                [newCat,~]=get_cat_membership(j);
                trainingData(j,4) = newCat; %save it in modelled category
            end
        end
        
        %% Do the whole loop of presenting instances
        
        function presentLoop(instances)
            instances = reshape(instances,1,length(instances));
            for i=instances
                presented = [presented i];
                if verbose>10
                    if mod(i,length(instances)/20) == 0
                        fprintf('.')
                    end
                end
                [newCat,~]=get_cat_membership(i);
                trainingData(i,4) = newCat;
                forget();
            end
        end
        
        %% Get log likelihood
        
        function [ll]=log_likelihood()
            % All data are presented now
            % Here we evaluate the test data.
            lens = testData(:,1);
            testData(:,2) = zeros(length(lens),1);% The version WITH
            % forgetting at the given forget rate
            catA = trainingData(trainingData(instances,4)==-1,1);
            catB = trainingData(trainingData(instances,4)==1,1);
            % just the lengths
            sumCatA=sum(exp(-choice_parameter*pdist2(catA, lens)));
            sumCatB=sum(exp(-choice_parameter*pdist2(catB, lens)));
            probA=sumCatA.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
            probB=sumCatB.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
            indices_a = find(testData(:,3)==-1); %ideal
            indices_b = find(testData(:,3)==1); %ideal
            testData(probA>probB,2) = -1;
            testData(probB>probA,2) = 1;
            ll=sum(log(probA(indices_a)));
            ll=ll+sum(log(probB(indices_b)));
            
            %% Below is without forgetting
            %catA = trainingData(trainingData(instances,5)==-1,1);
            %catB = trainingData(trainingData(instances,5)==1,1);
            %sumCatA=sum(exp(-choice_parameter*pdist2(catA, lens)));
            %sumCatB=sum(exp(-choice_parameter*pdist2(catB, lens)));
            %probA=sumCatA.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
            %probB=sumCatB.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
            %indices_a = find(testData(:,1)<30.5);
            %indices_b = find(testData(:,1)>30.5);
            %testData(probA>probB,3) = -1;
            %testData(probB>probA,3) = 1;
            %ll_no_forgetting = sum(log(probA(indices_a)))+sum(log(probB(indices_b)));
            
        end
        
        %% Get the required instance IDs and init presented instances
        instances = 1:noInstances;
        presented = [];
        %% Run the model
        
        presentLoop(instances);
        
        %% Compute the log-likelihood of test data
        
        if verbose>10
            disp(' ')
        end
        
        [ll]=log_likelihood();
        
        if verbose>10
            disp(' ')
        end
        
    end


%% loop through possibilities
feedback_types = [1 2]; %1- actual, 2- ideal
feedback_amounts = 1:11; %1- 100%, 2- some taken out
N_repeats = 100;

%% parameters
forget_rates = [0, 1E-7, 1E-5, 1E-3, 0.1, 0.2, 0.5];
ps_ids_used = [112101 112102 112103 112104 112105 112106 112107 ...
    112108 112110 112202 112203 112204 112205 112206 112207 112208 112209 ...
    112210 112211];

%% init files
fname_train = ['../../GCM_predictions/Gyslain/predictions_training.csv'];
fname_test = ['../../GCM_predictions/Gyslain/predictions_test.csv'];
ftrain = fopen(fname_train, 'w');
fprintf(ftrain, 'ps_id,forget_rate,length,feedback,ideal,model_forg\n', 1);
ftest = fopen(fname_test, 'w');
fprintf(ftest, 'ps_id,forget_rate,length,model_forg,ideal\n', 1);


for ps=ps_ids_used
    for frate = forget_rates
        [trainingset,testset] = read_training_stimuli(...
            '../../raw_data/randall_learn_recoded.csv',...
            '../../raw_data/randall_test_recoded.csv',ps);
        for rep = 1:N_repeats
            [train,ll,test] = GCM_generative_model('training_set', trainingset,...
                'test_set', testset, 'forget_rate', frate, 'ps_id', ps);
        end

        %% save the training data
        [nrows,~]= size(train);
        for row=1:nrows
            fprintf(ftrain, '%s,%.5f,%.2f,%d,%d,%d\n', ps,frate,train(row,:));
        end
        %% save the test data
        [nrows,~]= size(test);
        for row=1:nrows
            fprintf(ftest, '%s,%.5f,%d,%d,%.2f,%d,%d\n',ps,frate, test(row,:));
        end
    end
end
end
