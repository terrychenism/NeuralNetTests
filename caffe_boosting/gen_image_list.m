clear all; close all; clc

dd = dir('training');

ratio = 0.7;
class = 2;

train = [];
test = [];
for j = 3:length(dd)
    folder_name = ['training/', char(dd(j).name)];
    d = dir(strcat(folder_name,'/','*.jpg'));
    Ntake = length(d);
    new_name = [];
    for i = 1: Ntake
%         if j == 3 || j == 4
%             label = 1;
%         else
%             label = 0;
%         end
        label = j - 3;
%         im = imread([folder_name,'/',d(i).name]);
%         im = imresize(im,[48 48]);
%         imwrite(im, [folder_name,'/',d(i).name]);
        
       % new_name{i} = [folder_name,'/', d(i).name,' ',int2str(label)];  
       new_name{i} = [dd(j).name,'/', d(i).name,' ',int2str(label)];  
    end
    new_name = new_name';
    each_train = new_name(1:floor(Ntake*ratio), :);
    each_val = new_name(floor(Ntake*ratio)+1 : end, :);
    train =  [train; each_train];
    test =  [test; each_val];
end

new_name = new_name';

Randnidx = randperm(size(train,1)); 
train = train(Randnidx,:); 
train_list = char(train);

Randnidx = randperm(size(test,1)); 
test = test(Randnidx,:); 
test_list = char(test);


dlmwrite('train_list.txt',train_list,'delimiter','');
dlmwrite('val_list.txt',test_list,'delimiter','');
