function trainingData=GCM_model(fname, varargin)

%% Init the model's parameters
p=inputParser;
addRequired(p, 'training_fname');
addOptional(p, 'test_fname','test_ps1.xls');
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

%% Read the training data file
trainingData = xlsread(training_fname);
noInstances = length(trainingData(:,1));
trainingData(:,8) = 2*(trainingData(:,5)>30.5)-1; % -1 for cat A (short), 1 for cat B (long)
trainingData(:,5) = trainingData(:,5) + (noise_mu + noise_sigma.*randn(noInstances,1));
% add perceptual noise
trainingData = [trainingData trainingData(:,6)]; % copy the feedback to modelled category.
% (1)ps_id, (2)session, (3)feedType, (4)trial, (5)length, (6)tarCat,
% (7)respCat, (8)idealCat, (9)modelledCat
testData = xlsread(test_fname);
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

    function [cat,ll] = get_cat_membership(inst_no)
        cat=0;
        presentedData=trainingData(setdiff(presented, inst_no),:);
        len = trainingData(inst_no,5);
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
            sumCatA=sum(exp(-choice_parameter*sqrt((catA-len).^2)));
            sumCatB=sum(exp(-choice_parameter*sqrt((catB-len).^2)));
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

for i=1:noInstances
    presented = [presented i];
    newCat=get_cat_membership(i);
    trainingData(i,9) = newCat;
end

% forgotten instances: a=find(rand(1,len(presented))<0.00001)
% presented order: b=[a(1:3) a(5:end) a(4)]
end