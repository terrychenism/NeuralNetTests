function [pos, neg] = pascal_data(cls)

% Get training data from the PASCAL dataset.

globals; 
pascal_init;



  % positive examples from train+val
    pos = []; 
    numpos = 1; 
    num = 0;  
    [a,b,c,d,p,q,z] = textread('BTSD_testing_GTclear.txt','%s %f %f %f %f %d %d');     
    for i = 1:length(a);
        if mod(i,10)==0
            fprintf('%s: parsing positives: %d/%d\n', cls, i, length(a));
        end;
        t = a{numpos};
        if z(numpos) == 2 && t(2) == '1'
            num = num + 1;
            pos(num).im = a{numpos};
            pos(num).x1 = int16(b(numpos));
            pos(num).y1 = int16(c(numpos));
            pos(num).x2 = int16(d(numpos));
            pos(num).y2 = int16(p(numpos));
            
        end
        numpos = numpos + 1;
    end

  % negative examples from train (this seems enough!)
  ids = textread(sprintf(VOCopts.imgsetpath, 'train_smallset'), '%s');
  neg = [];
  numneg = 0;
  for i = 1:length(ids);
    fprintf('%s: parsing negatives: %d/%d\n', cls, i, length(ids));
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    if length(clsinds) == 0
      numneg = numneg+1;
      neg(numneg).im = [VOCopts.datadir rec.imgname];
    end
  end
  
  save('data_train', 'pos', 'neg');
 
