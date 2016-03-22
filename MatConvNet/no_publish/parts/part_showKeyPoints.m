function part_showKeyPoints(image, loc)
    figure
    
    imshow(image);
    hold on 
    pnow=loc;
    for i=1:15
       if pnow(3, i)
           plot([pnow(1,i)], [pnow(2, i)], 'r*', 'MarkerSize', 10);
       end
    end
    hold off
end