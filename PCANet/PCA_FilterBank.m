function V = PCA_FilterBank(InImg, PatchSize, NumFilters) 



ImgZ = length(InImg);
MaxSamples = 100000;
NumRSamples = min(ImgZ, MaxSamples); 

RandIdx = randperm(ImgZ);
RandIdx = RandIdx(1:NumRSamples);


NumChls = size(InImg{1},3);
Rx = zeros(NumChls*PatchSize^2,NumChls*PatchSize^2);

for i = RandIdx 
    im = im2col_general(InImg{i},[PatchSize PatchSize]); 
    im = bsxfun(@minus, im, mean(im)); 
    Rx = Rx + im*im'; 
end


Rx = Rx/(NumRSamples*size(im,2));

[E D] = eig(Rx);




[trash ind] = sort(diag(D),'descend');
V = E(:,ind(1:NumFilters)); 



 



