function [f V BlkIdx] = PCANet_train(InImg,PCANet,IdtExt)



NumImg = length(InImg);

V = cell(PCANet.NumStages,1); 
OutImg = InImg; 
ImgIdx = (1:NumImg)';
clear InImg; 

for stage = 1:PCANet.NumStages
    display(['Computing PCA filter bank and its outputs at stage ' num2str(stage) '...'])
    
    V{stage} = PCA_FilterBank(OutImg, PCANet.PatchSize, PCANet.NumFilters(stage)); % compute PCA filter banks
    
    if stage ~= PCANet.NumStages % compute the PCA 
        [OutImg ImgIdx] = PCA_output(OutImg, ImgIdx, ...
            PCANet.PatchSize, PCANet.NumFilters(stage), V{stage});  
    end
end

if IdtExt == 1     
    f = cell(NumImg,1); 
    
    for idx = 1:NumImg
        if 0==mod(idx,100); display(['Extracting PCANet feasture of the ' num2str(idx) 'th training sample...']); end
        OutImgIndex = ImgIdx==idx; 
        
        [OutImg_i ImgIdx_i] = PCA_output(OutImg(OutImgIndex), ones(sum(OutImgIndex),1),...
            PCANet.PatchSize, PCANet.NumFilters(end), V{end});  % compute  image "idx"
        
        [f{idx} BlkIdx] = HashingHist(PCANet,ImgIdx_i,OutImg_i); % compute the feature of image(idx)
        OutImg(OutImgIndex) = cell(sum(OutImgIndex),1); 
    end
    f = [f{:}];

end







