function loc=part_original2reshape(loc, hw)
%% map an part location from original coordinate system to a new one
    % loc is a 3*15 location matrix
    sz=448; % size after the reshape
    factor=max(sz/hw(1), sz/hw(2));
    loc(1:2, :)=factor*loc(1:2, :);

    hw=factor*hw;
    shift=(hw-sz)/2;

    loc(1:2, :)=bsxfun(@minus, loc(1:2,:), shift);
    
    % some keypoints might not be visible any more
    isvis=(loc(1,:)<=sz) & (loc(1,:)>=1) & ...
          (loc(2,:)<=sz) & (loc(2,:)>=1);
    loc(3, ~isvis)=0;
end