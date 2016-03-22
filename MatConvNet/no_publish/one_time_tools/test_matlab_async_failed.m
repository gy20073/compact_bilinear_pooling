function test2()
% test the ability of matlab async calls
% the conclusion is Matlab couldn't do async call
% except when the time of execution is significantly
% larger than the call overhead(3 second - 30 second)

%%
t=timer;
t.TimerFcn='ready=ready+1; b=33;';
start(t)

ready
end


 global ready next_curImages next_labels curIds smallFcn
    smallFcn=@(ids)getBatchFunc(imdb, ids);
    % start a reading process
    t=timer;
    curIds=batchIds(getInter(1, batchSize, numel(batchIds)));
    t.TimerFcn='global ready next_curImages next_labels curIds smallFcn; ready=0;[next_curImages, next_labels]=smallFcn(curIds); ready=1;';
    start(t);


if 1
while ready~=1
    sleep(1);
    fprintf('wait for getBatch\n');
end
% wait ends
curImages=next_curImages;
labels=next_labels;
curIds=nextInter;
% start the timer for the next batch
t=timer;
t.TimerFcn='global ready next_curImages next_labels curIds smallFcn; ready=0;[next_curImages, next_labels]=smallFcn(curIds); ready=1;';
tic
start(t);
toc
end