function GCM_generate_Ps_responses(forget_rate)

    function [training,test] = get_training_stimuli(feed_type, feed_amount)
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
        
        %% Remove feedback if needed
        if feed_amount > 1
            %if partial feedback then select some randome elements of A
            no_ommitted = 8*(feed_amount-1);
            rand_A = randperm(80,no_ommitted);
            %multiply them by two so that we know which distribution it came from.
            list_A(rand_A,2) = list_A(rand_A,2)*2;
            %same for category D
            rand_D = randperm(80,no_ommitted);
            list_D(rand_D,2) = list_D(rand_D,2)*2;
        end
        
        %% Add test set
        list_test(:,2) = (list_test>30.5) * 2 - 1;
        %% shuffle both lists
        list_A=list_A(randperm(length(list_A)),:);
        list_D=list_D(randperm(length(list_D)),:);
        list_test = list_test(randperm(length(list_test)),:);
        
        training = [list_A;list_D];
        test=list_test;
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
            fprintf('Model with gamma %.1f, forget rate %.10f, choice parameter %d, noise mean %.1f, noise sd %.1f.\n',...
                gamma, forget_rate,choice_parameter,noise_mu,noise_sigma);
        end
        
        %% Read the training and test data file
        
        trainingData = training_set;
        noInstances = length(trainingData(:,1));
        trainingData(:,3) = (trainingData(:,1)>30.5) * 2 - 1;
        % ideal feedback
        trainingData(:,1) = trainingData(:,1) + (noise_mu + noise_sigma.*randn(noInstances,1));
        % add perceptual noise
        trainingData = [trainingData trainingData(:,2)];
        % copy the feedback to modelled category (twice, once for without
        % and once for with forgetting).
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
            presentedData=trainingData(presented(1:end-1),:);
            len = trainingData(inst_no,1);
            oldCat = trainingData(inst_no,4); %using modelled category here,
            % it is the same as presented category to begin with anyway.
            if (oldCat == -2) || (oldCat == 2)
                % no feedback was given
            end
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
N_per_cell = 1000;
N_repeats = 1;
fname_train = ['../GCM_predictions/predictions_training' num2str(forget_rate) '.csv'];
fname_test = ['../GCM_predictions/predictions_test' num2str(forget_rate) '.csv'];
ftrain = fopen(fname_train, 'w');
if forget_rate == 0
    fprintf(ftrain, 'ps_id,forget_rate,feedback_type,feedback_amount,length,feedback,ideal,model_forg\n', 1);
end
ftest = fopen(fname_test, 'w');
if forget_rate==0
    fprintf(ftest, 'ps_id,forget_rate,feedback_type,feedback_amount,length,model_forg,ideal\n', 1);
end
for fType = feedback_types
    for fAmount = feedback_amounts
        for ps=1:N_per_cell
            [trainingset,testset] = get_training_stimuli(fType, fAmount);
            for rep = 1:N_repeats
                [train,ll,test] = GCM_generative_model('training_set', trainingset,...
                    'test_set', testset, 'forget_rate', forget_rate);
            end
            ps_id = sprintf('%1d%1d%02d', fType,fAmount,ps);
            
            %% save the training data
            [nrows,~]= size(train);
            for row=1:nrows
                fprintf(ftrain, '%s,%.5f,%d,%d,%.2f,%d,%d,%d\n', ps_id,forget_rate,fType,fAmount,train(row,:));
            end
            %% save the test data
            [nrows,~]= size(test);
            for row=1:nrows
                fprintf(ftest, '%s,%.5f,%d,%d,%.2f,%d,%d\n',ps_id,forget_rate,fType,fAmount, test(row,:));
            end
        end
    end
end
end
