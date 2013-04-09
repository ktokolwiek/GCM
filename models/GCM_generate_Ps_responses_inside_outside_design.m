function GCM_generate_Ps_responses_inside_outside_design(overlap)

    function [training,test] = get_training_stimuli(feed_type)
        % the lengths are divided into 6 regions - 100% cat A, 60% cat A,
        % blank, blank, 40% cat A, 0% cat A
        % phase 1 - we train on regions 1 and 6.
        training_phase_1 = [repmat((1:10)',2,1); repmat((51:60)',2,1)];
        training_phase_1 = shuffle(training_phase_1);
        training_phase_1(:,2) = (training_phase_1>30)*2-1;
        training_phase_2 = [repmat((1:20)',2,1); repmat((41:60)',2,1)];
        training_phase_2(training_phase_2(:,1)<11,2) = -1;
        training_phase_2(training_phase_2(:,1)>49,2) = 1;
        if feed_type == 2
            % if we give Ps the 60% / 40% feedback in regions 2 and 5
            multiplier = 1;
        else
            multiplier = 2; % means we give Ps the 'fake' feedback
        end
        for i=1:length(training_phase_2)
            if training_phase_2(i,2) == 0
                % i.e. it is in region 3 or 5
                if training_phase_2(i,1) <30
                    %it is 60% cat A
                    prob = 0.6;
                else
                    % it is 40% cat A
                    prob = 0.4;
                end
                if randn < prob
                    training_phase_2(i,2) = -multiplier;
                else
                    training_phase_2(i,2) = multiplier;
                end
            end
        end
        list_test = (1:60)';
        %% Add test set
        list_test(:,2) = (list_test>30.5) * 2 - 1;
        %% shuffle both lists (training list order is already found in order_of_presentation.
        test = list_test(randperm(length(list_test)),:);
        training_phase_2 = training_phase_2(randperm(length(training_phase_2)),:);
        training = [training_phase_1; training_phase_2];
    end


%%%%%%%%
%% Here we need to implement the different feedback types and how we infer
% from the length.
%% .

    function [trainingData, testData, ll_train, ll_test] = GCM_generative_model(varargin)
        
        %% Init the model's parameters
        p=inputParser;
        addOptional(p, 'training_set', NaN);
        addOptional(p, 'test_set', NaN);
        addOptional(p, 'verbose',0,@isnumeric);
        addOptional(p, 'gamma',2,@isnumeric);
        addOptional(p, 'forget_rate',0.05,@isnumeric);
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
                gamma, choice_parameter,noise_mu,noise_sigma);
        end
        
        %% Read the training and test data file
        
        trainingData = training_set;
        noInstances = length(trainingData(:,1));
        trainingData(:,3) = (trainingData(:,1)>30.5) * 2 - 1;
        % ideal feedback
        trainingData(:,1) = trainingData(:,1) + (noise_mu + noise_sigma.*randn(noInstances,1));
        % add perceptual noise
        trainingData(:,4) = zeros(noInstances,1);
        trainingData = [trainingData trainingData(:,2) trainingData(:,2)];
        % copy the feedback to modelled category (twice, once for without
        % and once for with forgetting).
        % (1) length, (2) feedback, (3) idealCat, (4)
        % modelledCat, i.e. the Ps response., (5) the actually received
        % feedback (6) state of memory (incl. forgetting)
        testData = test_set;
        testData(:,3) = (testData(:,1)>30.5) * 2 - 1;
        testData(:,1) = testData(:,1) + (noise_mu + noise_sigma.*randn(length(testData(:,1)),1));
        % add perceptual noise
        testData(:,2) = zeros(size(testData(:,1)));
        % (1) length, (2) forgetting, (3) ideal
        
        
        %% Get category memberships
        
        function [cat,ll] = get_cat_membership(inst_no)
            presentedData=trainingData(presented(1:end-1),:); % We don't have
            % the feedback for the instamnce which is just presented (the
            % one which we are comparing now to all the previously
            % presented ones).
            len = trainingData(inst_no,1);
            % we use the sixth column, i.e. the memory state, incl.
            % feedback we actually received (including the 'fake' feedback).
            catA = presentedData(presentedData(:,6)==-1,1);
            catB = presentedData(presentedData(:,6)==1,1);
            % just the lengths of instances in Cat A or B
            if isempty(catA) || isempty(catB)
                % if we haven't seen at least one instance of both
                % categories, then we have to guess with probability of
                % 0.5.
                ll=log(0.5);
                if rand<0.5
                    cat =  -1;
                else
                    cat = 1;
                end
            else
                sumCatA=sum(exp(-choice_parameter*abs(catA-len)));
                sumCatB=sum(exp(-choice_parameter*abs(catB-len)));
                probA=sumCatA^gamma/(sumCatA^gamma+sumCatB^gamma);
                probB=sumCatB^gamma/(sumCatA^gamma+sumCatB^gamma);
                if probA>probB
                    cat = -1;
                else
                    cat = 1;
                end
                %% We get the log prob of the ideal category.
                idealCat = trainingData(inst_no,3);
                if idealCat == -1
                    ll = log(probA);
                else
                    ll = log(probB);
                end
            end
        end
              
        %% Forget instance
        function forget()
            presented_before = presented;
            for inst=presented_before
                if rand<forget_rate
                    % remove this inst from presented
                    presented = setdiff(presented, inst);
                    [cat,~] = get_cat_membership(inst);
                    presented = [presented inst];
                    trainingData(inst,6)=cat; % save it in the memory state.
                end
            end
        end
        
        %% Do the whole loop of presenting instances
        
        function log_prob=presentLoop(instances)
            instances = reshape(instances,1,length(instances));
            log_prob = 0;
            for i=instances
                presented = [presented i];
                if verbose>10
                    if mod(i,length(instances)/20) == 0
                        fprintf('.')
                    end
                end
                [newCat,logprob]=get_cat_membership(i); % that estimates the Ps
                % response and if no feedback is presented, then that
                % response is saved as feedback for that instance.
                log_prob=log_prob+logprob;
                trainingData(i,4) = newCat; % save the response.
                %% Now we see whether we got feedback for this curent instance,
                % or whether we need to present the response as feedback.
                oldCat = trainingData(i,2);
                if (oldCat == -2) || (oldCat == 2)
                    % no feedback was given, so the feedback is whatever we
                    % find here. Save it as the feedback received and as
                    % the memory state.
                    trainingData(i,5:6) = newCat;
                end
                forget();
            end
        end
        
        %% Get log likelihood
        
        function ll=log_likelihood()
            % All data are presented now
            % Here we evaluate the test data.
            lens = testData(:,1);
            testData(:,2) = zeros(length(lens),1);
            % We use column 6 here, which is the memory state. It includes 
            % feedback we actually received, including the self-generated 
            % feedback.
            catA = trainingData(trainingData(instances,6)==-1,1);
            catB = trainingData(trainingData(instances,6)==1,1);
            % just the lengths
            sumCatA=sum(exp(-choice_parameter*pdist2(catA, lens)));
            sumCatB=sum(exp(-choice_parameter*pdist2(catB, lens)));
            probA=sumCatA.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
            probB=sumCatB.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
            indices_a = find(testData(:,3)==-1); %ideal
            indices_b = find(testData(:,3)==1); %ideal
            testData(probA>probB,2) = -1; %save the Ps response
            testData(probB>probA,2) = 1;
            ll=sum(log(probA(indices_a)));
            ll=ll+sum(log(probB(indices_b)));            
        end
        
        %% Get the required instance IDs and init presented instances
        instances = 1:noInstances;
        presented = [];
        %% Run the model
        
        ll_train=presentLoop(instances);
        
        %% Compute the log-likelihood of test data
        
        if verbose>10
            disp(' ')
        end
        
        ll_test=log_likelihood();
        
        if verbose>10
            disp(' ')
        end
        
    end



%% loop through possibilities
feedback_types = [1 2]; %1- 'fake', 2- 60%/40%
N_per_cell = 1000;
N_repeats = 1;
fname_train = '../GCM_predictions/predictions_training_study_design.csv';
fname_test = '../GCM_predictions/predictions_test_study_design.csv';
ftrain = fopen(fname_train, 'w');
fprintf(ftrain, 'ps_id,feedback_type,length,feedback,ideal,model_answer,feedback_actually_received,memory_state\n', 1);
ftest = fopen(fname_test, 'w');
fprintf(ftest, 'ps_id,feedback_type,length,model_answer,ideal\n', 1);
for fType = feedback_types
    for ps=1:N_per_cell
        [trainingset,testset] = get_training_stimuli(fType);
        for rep = 1:N_repeats
            [train,test,ll_train,ll_test] = GCM_generative_model('training_set', trainingset,...
                'test_set', testset);
        end
        ps_id = sprintf('%1d%02d', fType,ps);
        
        %% save the training data
        [nrows,~]= size(train);
        for row=1:nrows
            fprintf(ftrain, '%s,%d,%.2f,%d,%d,%d,%d,%d\n', ps_id,fType,train(row,:));
        end
        %% save the test data
        [nrows,~]= size(test);
        for row=1:nrows
            fprintf(ftest, '%s,%d,%.2f,%d,%d\n',ps_id,fType,test(row,:));
        end
        
    end
end
end
