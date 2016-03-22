function mAP=score_mAP(classifications, testY)
    nclass=max(testY);
    mAP=zeros(1, nclass);
    for c=1:nclass
        testy=2*(testY==c)-1;
        [~,~,i]= vl_pr(testy, classifications(:,c)); 
        mAP(c) = i.ap ; 
    end
end