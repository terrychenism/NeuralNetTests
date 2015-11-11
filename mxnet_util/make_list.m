[x, y] = textread('train.txt','%s %s');

val_list  = [];

for i = 1:size(x,1)
    val_list = [val_list; strcat(int2str(i),{' '}, y(i), {' '} , x(i))];
end 

Randnidx = randperm(size(val_list,1)); 
val_list = val_list(Randnidx,:); 
test_list = char(val_list);

dlmwrite('train_list.txt',test_list,'delimiter','');
