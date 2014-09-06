function names = getImageSet(path)


content = dir(path) ;
names = {content.name} ;
ok = regexpi(names, '.*\.(jpg|png|jpeg|gif|bmp|tiff)$', 'start') ;
names = names(~cellfun(@isempty,ok)) ;

for i = 1:length(names)
  names{i} = fullfile(path,names{i}) ;
end
