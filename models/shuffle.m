function shuffled=shuffle(list)
%% Returns a randomly shuffled version of the list (vector, not matrix).
shuffled = list(randperm(length(list)));
end