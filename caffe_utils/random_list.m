predict_file='categories.txt';
[predlist, preds{1}] = textread(predict_file, '%s %d');
fileID = fopen('names.txt','w');
for i = 1:length(predlist)
    origstr = predlist{i};
    newstr = strrep(origstr, '/', ' ');
    newstr = strrep(newstr, '_', ' ');
    predlist{i,2} = newstr(3:end);  
    fprintf(fileID,'%s\n',newstr(3:end));
end
fclose(fileID);
save('predlist.mat','predlist');
