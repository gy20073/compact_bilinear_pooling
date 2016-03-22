function imdb=modifyImdb(imdb)
%%
    classes=imdb.classes.name;
    
    id=[];
    name={};
    label=[];
    set=[];
    
    iset=1;
    iid=1;
    
    for i=1 : numel(classes)
        class=classes{i};
        path=fullfile(imdb.imageDir, class);
        files=dir(path);
        for j=1:numel(files)
            if ~files(j).isdir
                id(end+1)=iid;
                name{end+1}=fullfile(class, files(j).name);
                label(end+1)=i;
                set(end+1)=iset;
                
                iset=3-iset;
                iid=iid+1;
            end
        end
    end
    
    out=[];
    out.id=id;
    out.name=name;
    out.label=label;
    out.set=set;
    
    imdb.images=out;
end