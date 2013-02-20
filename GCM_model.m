function [trainingData, ll] =GCM_model(fname, varargin)

%% Init the model's parameters
p=inputParser;
addRequired(p, 'training_fname');
addOptional(p, 'test_fname','test_all.xls');
addOptional(p, 'verbose',15,@isnumeric);
addOptional(p, 'gamma',1,@isnumeric);
addOptional(p, 'forget_rate',0.00001,@isnumeric);
addOptional(p, 'choice_parameter', 1, @isnumeric);
addOptional(p, 'noise_mu',0,@isnumeric);
addOptional(p, 'noise_sigma',0.5, @isnumeric);
parse(p, fname, varargin{:})
training_fname = p.Results.training_fname;
test_fname = p.Results.test_fname;
verbose = p.Results.verbose;
gamma = p.Results.gamma;
forget_rate = p.Results.forget_rate;
choice_parameter = p.Results.choice_parameter;
noise_mu = p.Results.noise_mu;
noise_sigma = p.Results.noise_sigma;

%% Read the training and test data file
trainingData = xlsread(training_fname);
noInstances = length(trainingData(:,1));
trainingData(:,8) = 2*(trainingData(:,5)>30.5)-1; % -1 for cat A (short), 1 for cat B (long)
trainingData(:,5) = trainingData(:,5) + (noise_mu + noise_sigma.*randn(noInstances,1));
% add perceptual noise
trainingData = [trainingData trainingData(:,6)]; % copy the feedback to modelled category.
% (1)ps_id, (2)session, (3)feedType, (4)trial, (5)length, (6)tarCat,
% (7)respCat, (8)idealCat, (9)modelledCat
testData = xlsread(test_fname);
testData(:,5) = testData(:,5) + (noise_mu + noise_sigma.*randn(length(testData(:,1)),1));
% (1)subj, (2)session, (3)feedType, (4)trial, (5)length, (6)respSE, 
% (7)respRT
%% Get indices of selected instances
    function [selInsts] = get_indices(varargin)
        p=inputParser;
        addOptional(p, 'dataset', 'training')
        addOptional(p, 'ps_id',NaN);
        addOptional(p, 'session',NaN);
        addOptional(p, 'feedType',NaN);
        addOptional(p, 'trial',NaN);
        parse(p, varargin{:})
        if strcmp(p.Results.dataset,'training')
            dataset = trainingData;
        else
            dataset = testData;
        end
        selInsts = 1:length(dataset(:,1));
        if ~isnan(p.Results.ps_id)
            selInsts=intersect(selInsts, find(dataset(:,1)==p.Results.ps_id));
        end
        if ~isnan(p.Results.session)
            selInsts=intersect(selInsts, find(dataset(:,2)==p.Results.session));
        end
        if ~isnan(p.Results.feedType)
            selInsts=intersect(selInsts, find(dataset(:,3)==p.Results.feedType));
        end
        if ~isnan(p.Results.trial)
            % this is a range
            selInsts=intersect(selInsts, find(ismember(dataset(:,4),p.Results.trial)));
        end
    end
presented = [];
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
    function ll=log_likelihood()
        % All data are presented now
        lens = testData(:,5);
        catA = trainingData(trainingData(instances,9)==-1,5);
        catB = trainingData(trainingData(instances,9)==1,5);
        % just the lengths
        sumCatA=sum(exp(-choice_parameter*pdist2(catA, lens)));
        sumCatB=sum(exp(-choice_parameter*pdist2(catB, lens)));
        probA=sumCatA.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
        probB=sumCatB.^gamma./(sumCatA.^gamma+sumCatB.^gamma);
        ll=sum(log(probA(find(testData(:,6)==-1))));
        ll=ll+sum(log(probB(find(testData(:,6)==1))));
    end

%% Run the model


instances = find(trainingData(:,1)==112101);
instances = 1:noInstances;
instances = reshape(instances,1,length(instances));

presentLoop(instances);

ll=0;
if verbose>10
    disp(' ')
end

% 
% for k=instances
%     if verbose>10
%         if mod(k,length(instances)/20) == 0
%             fprintf('.')
%         end
%     end
%     [~,x]=get_cat_membership(k);
%     ll=ll+x;
% end

ll=log_likelihood();

if verbose>10
    disp(' ')
end

% forgotten instances: a=find(rand(1,len(presented))<0.00001)
% presented order: b=[a(1:3) a(5:end) a(4)]
end