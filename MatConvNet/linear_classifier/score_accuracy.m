function accuracy=score_accuracy(classifications, testY)
    [~, pre]=max(classifications, [],2);
    accuracy=1.0*sum(pre==testY)/numel(testY);
end