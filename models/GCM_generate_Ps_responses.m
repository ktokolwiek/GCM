function GCM_generate_Ps_responses()

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
        if feed_amount == 2
            %if partial feedback then select some randome elements of A
            rand_A = randperm(80,25);
            %multiply them by two so that we know which distribution it came from.
            list_A(2,rand_A) = list_A(2,rand_A)*2;
            %same for category D
            rand_D = randperm(80,25);
            list_D(2,rand_D) = list_D(2,rand_D)*2;
        end
        
        %% Add test set
        list_test = (list_test>30.5) * 2 - 1;
        
        %% shuffle both lists
        list_A=list_A(randperm(length(list_A)),:);
        list_D=list_D(randperm(length(list_D)),:);
        list_test = list_test(randperm(length(list_test)));
        
        training = [list_A;list_D];
        test=list_test;
    end



%%%%%%%%
%% Here we need to implement the different feedback types and how we infer
% from the length.
%% .
function [trainingData,ll, testData, ll_no_forgetting] = GCM_model(varargin)

%% Init the model's parameters
p=inputParser;
addOptional(p, 'training_set', NaN);
addOptional(p, 'test_set', NaN);
addOptional(p, 'verbose',15,@isnumeric);
addOptional(p, 'gamma',1,@isnumeric);
addOptional(p, 'forget_rate',0.00001,@isnumeric);
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
trainingData(:,1) = trainingData(:,1) + (noise_mu + noise_sigma.*randn(noInstances,1));
% add perceptual noise
trainingData = [trainingData trainingData(:,2)]; % copy the feedback to modelled category.
% (1) length, (2) feedback
% (1)ps_id, (2)session, (3)feedType, (4)trial, (5)length, (6)tarCat,
% (7)respCat, (8)idealCat, (9)modelledCat_with_forgetting, (10)modelledCat_no_forgetting
testData = xlsread(test_fname);
testData(:,5) = testData(:,5) + (noise_mu + noise_sigma.*randn(length(testData(:,1)),1));
% add perceptual noise
% (1)subj, (2)session, (3)feedType, (4)trial, (5)length, (6)respSE, 
% (7)respRT (8)modelledCat (9)modelledCat_no_forgetting

%% Get indices of selected instances
  
    function [trainingInsts, testInsts] = get_indices()
        trainingInsts = 1:length(trainingData(:,1));
        testInsts = 1:length(testData(:,1));
        if ~isnan(ps_id)
            trainingInsts=intersect(trainingInsts, find(trainingData(:,1)==ps_id));
            testInsts=intersect(testInsts, find(testData(:,1)==ps_id));
        end
        if ~isnan(session)
            trainingInsts=intersect(trainingInsts, find(trainingData(:,2)==session));
        end
        if ~isnan(feedType)
            trainingInsts=intersect(trainingInsts, find(trainingData(:,3)==feedType));
        end
        if ~isnan(trial)
            % this is a range
            trainingInsts=intersect(trainingInsts, find(ismember(trainingData(:,4),trial)));
            testInsts=intersect(testInsts, find(ismember(testData(:,4),trial)));
        end
    end

%% Get category memberships
  
    function [cat,ll] = get_cat_membership(inst_no)
        presentedData=trainingData(presented,:);
        len = trainingData(inst_no,5);
        oldCat = trainingData(inst_no,9);
        catA = presentedData(presentedData(:,9)==-1,5);
        catB = presentedData(presentedData(:,9)==1,5);
        % just the lengths
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
            if oldCat == -1
                sumCatA = sumCatA-1;
            else
                sumCatB = sumCatB-1;
            end % we subtract exp(0) for difference with itself.
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
            trainingData(j,9) = newCat;
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
            trainingData(i,9) = newCat;
            forget();
        end
    end

%% Get log likelihood

    function [ll, ll_no_forgetting]=log_likelihood()
        % All data are presented now
        lens = testData(:,5);
        testData(:,8) = zeros(length(lens),1);
        testData(:,9) = zeros(length(lens),1); % The version WITHOUT forgetting
        catA = trainingData(trainingData(instances,9)==-1,5);
        catB = trainingData(trainingData(instances,9)==1,5);
        % just the lengths
        sumCatA=sum(exp(-choice_parameter*pdist2(catA, lens)));
        sumCatB=sum(exp(-choice_parameter*pdist2(catB, lens)));
        probA=sumCatA.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
        probB=sumCatB.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
        indices_a = intersect(testInstances, find(testData(:,6)==-1));
        indices_b = intersect(testInstances, find(testData(:,6)==1));
        testData(probA>probB,8) = -1;
        testData(probB>probA,8) = 1;
        ll=sum(log(probA(indices_a)));
        ll=ll+sum(log(probB(indices_b)));

        % Below is without forgetting
        catA = trainingData(trainingData(instances,8)==-1,5);
        catB = trainingData(trainingData(instances,8)==1,5);
        sumCatA=sum(exp(-choice_parameter*pdist2(catA, lens)));
        sumCatB=sum(exp(-choice_parameter*pdist2(catB, lens)));
        probA=sumCatA.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
        probB=sumCatB.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
        indices_a = intersect(testInstances, find(testData(:,6)==-1));
        indices_b = intersect(testInstances, find(testData(:,6)==1));
        testData(probA>probB,9) = -1;
        testData(probB>probA,9) = 1;
        ll_no_forgetting = sum(log(probA(indices_a)))+sum(log(probB(indices_b)));
        
    end

%% Get the required instance IDs and init presented instances

[instances, testInstances] = get_indices();
presented = [];

%% Run the model

presentLoop(instances);
a=get_ideal_membership(instances);

%% Compute the log-likelihood of test data

if verbose>10
    disp(' ')
end

[ll, ll_no_forgetting]=log_likelihood();

if verbose>10
    disp(' ')
end

end


%% loop through possibilities
feedback_types = [1 2]; %1- actual, 2- ideal
feedback_amounts = [1 2]; %1- 100%, 2- some taken out
N_per_cell = 50;
N_repeats = 1;

for fType = feedback_types
   for fAmount = feedback_amounts
       train_data_all = [];
       test_data_all = [];
       for ps=1:N_per_cell
           for rep = 1:N_repeats
               [train,test] = GCM_generative_model();
           end
           test_data_all = [test_data_all; test];
           train_data_all = [train_data_all; train];
       end
   end
end




end