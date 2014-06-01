%% =========== txt2cvs ======
load('out.txt');
filename_submission = 'submission.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','ImageId','Label');
for i = 1 : length(out)
    fprintf(f,'%d,%d\n',i,out(i));
end
fclose(f);
disp('Done.');
