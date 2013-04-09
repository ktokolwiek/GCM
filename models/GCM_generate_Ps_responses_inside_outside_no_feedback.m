function GCM_generate_Ps_responses_inside_outside_design(overlap)

    function [training,test] = get_training_stimuli(feed_type, feed_amount, region)
        % region 1 = overlapping (right half of the left distribution and
        % left half of the right distribution)
        % region 2 = non-overlapping (left half of the left distribution
        % and right half of the right distribution)
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
        N_first_instances = 5; % minimum number of instances with feedback at the beginning
        %% Take the N first instances randomly sampled from both distributions
        first_A = randperm(80,N_first_instances);
        first_D = randperm(80,N_first_instances)+80;
        indices_first = [first_A first_D];
        indices_first = shuffle(indices_first); % shuffle the indices
        if region == 1
            % overlapping
            no_feedback_indices_A = setdiff(81-overlap:80, first_A);
            feedback_indices_A = setdiff(1:80-overlap, first_A);
            no_feedback_indices_D = setdiff(81:80+overlap, first_D);
            feedback_indices_D = setdiff(81+overlap:160, first_D);
        elseif region == 2
            % non-overlapping
            no_feedback_indices_A = setdiff(1:80-overlap, first_A);
            feedback_indices_A = setdiff(81-overlap:80, first_A);
            no_feedback_indices_D = setdiff(81+overlap:160, first_D);
            feedback_indices_D = setdiff(81:80+overlap, first_D);
        end
        
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
        list_training = [list_A; list_D]; % list with all training instances.
        
        %% Which feedback to remove?
        % Make sure here that the lists are shuffled.
        if feed_amount > 1
            %if partial feedback then select some randome elements of A
            no_ommitted_A = ceil(length(no_feedback_indices_A)/10*(feed_amount-1));
            no_ommitted_D = ceil(length(no_feedback_indices_D)/10*(feed_amount-1));
            % ^ count how many
            rand_A = randsample(no_feedback_indices_A,no_ommitted_A);
            rand_D = randsample(no_feedback_indices_D,no_ommitted_D);
            % ^ sample from the lists of feedback to remove
        else
            % no feedback is removed, so these indices are empty.
            rand_A=[];
            rand_D=[];
        end
        % Now we know for which indices we remove feedback (rand_A and
        % rand_D) and for which ones we don't (feedback_indices_A and
        % feedback_indices_D). Just shuffle the indices and build a list,
        % which is order of presentation.
        feedback_indices_A = [feedback_indices_A setdiff(no_feedback_indices_A,rand_A)];
        feedback_indices_D = [feedback_indices_D setdiff(no_feedback_indices_D,rand_D)];
        indices_later = [feedback_indices_A feedback_indices_D rand_A rand_D];
        indices_later = shuffle(indices_later);
        order_of_presentation = [indices_first indices_later];
        %% Now remove the feedback for the instances we found
        list_training([rand_A rand_D],2) = list_training([rand_A rand_D],2)*2; % mark it times 2
        %% Add test set
        list_test(:,2) = (list_test>30.5) * 2 - 1;
        %% shuffle both lists (training list order is already found in order_of_presentation.
        test = list_test(randperm(length(list_test)),:);
        training = list_training(order_of_presentation,:);
    end


%%%%%%%%%%%%%%%% edit 4/04/2013 up to here.
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
                    % no feedback was given,
                    % so we put it in neither category.
                    trainingData(i,5:6) = 0;
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
feedback_types = [1 2]; %1- actual, 2- ideal
feedback_amounts = 1:11; %1- 100%, 2- some taken out
feedback_removed = [1 2]; %1- overlap, 2-no overlap
N_per_cell = 100;
N_repeats = 1;
fname_train = '../GCM_predictions/predictions_training_overlap_20_no_feed.csv';
fname_test = '../GCM_predictions/predictions_test_overlap_20_no_feed.csv';
ftrain = fopen(fname_train, 'w');
fprintf(ftrain, 'ps_id,feedback_type,feedback_amount,region,length,feedback,ideal,model_answer,feedback_actually_received,memory_state\n', 1);
ftest = fopen(fname_test, 'w');
fprintf(ftest, 'ps_id,feedback_type,feedback_amount,region,length,model_answer,ideal\n', 1);
for fType = feedback_types
    for fAmount = feedback_amounts
        for region = feedback_removed
            for ps=1:N_per_cell
                [trainingset,testset] = get_training_stimuli(fType, fAmount, region);
                for rep = 1:N_repeats
                    [train,test,ll_train,ll_test] = GCM_generative_model('training_set', trainingset,...
                        'test_set', testset);
                end
                ps_id = sprintf('%1d%1d%1d%02d', fType,fAmount,region,ps);
                
                %% save the training data
                [nrows,~]= size(train);
                for row=1:nrows
                    fprintf(ftrain, '%s,%d,%d,%d,%.2f,%d,%d,%d,%d,%d\n', ps_id,fType,fAmount,region,train(row,:));
                end
                %% save the test data
                [nrows,~]= size(test);
                for row=1:nrows
                    fprintf(ftest, '%s,%d,%d,%d,%.2f,%d,%d\n',ps_id,fType,fAmount,region,test(row,:));
                end
            end
        end
    end
end
end
