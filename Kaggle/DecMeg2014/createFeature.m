clear

XX(:, :, 1)= [ 1 2 3; 4 5 6; 7 8 9];
XX(:, :, 2)= [ 11 12 13; 14 15 16; 17 18 19];

features = single(zeros(size(XX,1),size(XX,2)*size(XX,3)));
for i = 1 : size(XX,1)
    temp = squeeze(XX(i,:,:))';
    features(i,:) = temp(:); 
end
