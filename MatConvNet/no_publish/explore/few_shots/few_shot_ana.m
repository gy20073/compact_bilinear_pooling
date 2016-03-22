% analyze the result of few shots learning
function few_shot_ana(out, dim)
    strDim=['n' num2str(dim)];

    shots=[1, 2, 3, 4, 5, 7, 9, 11, 14];
    bl_map=[];
    c2_map=[];
    for shot=shots
        name=['n' num2str(shot)];
        t=out.(strDim).(name);
        bl_map=[bl_map out.n8192.(name).bilinear{2}];
        c2_map=[c2_map t.compact2{2}];
    end

    bl_map
    c2_map

    plot(shots, bl_map, '*-', shots, c2_map, '+-');
    xlabel('shots for each class');
    ylabel('mAP');
    legend({'bilinear', 'compact tensor'})
end