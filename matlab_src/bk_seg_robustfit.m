function bk_seg_robustfit(startim_id, endim_id)
run('/home/boyu/env/matlabconfig.m')
dataDir = '../data/2D_materials/Graphene and Graphite';
saveDir = '../results/bk_seg_robustfit';
img_names = dir([dataDir, '/0*.jpg']);
num_imgs = length(img_names);
for im_id = startim_id: endim_id
    ml_progressBar(im_id, endim_id);
    %im = imread(sprintf('%s/07032018 G chip7-000007.jpg', dataDir));
    im = imread(sprintf('%s/%s', dataDir, img_names(im_id).name));
    im = imresize(im, 0.25);
    im_hsv = rgb2hsv(im);
    im_gray = rgb2gray(im);
%     im_c = double(im_gray(:,:,1));

    mask = im_gray(:) > 20; % pixels which are not completely black. The mask is to avoid black border.
    
%     outlier_map = perform_robustfit(double(im_gray(:,:,1)), mask);
%     imwrite(outlier_map, fullfile(saveDir, [img_names(im_id).name(1:end-4) '_gray.png']))
%     outlier_map = perform_robustfit(double(im_hsv(:,:,1)), mask);
%     imwrite(outlier_map, fullfile(saveDir, [img_names(im_id).name(1:end-4) '_hue.png']))
%     outlier_map = perform_robustfit(double(im_hsv(:,:,2)), mask);
%     imwrite(outlier_map, fullfile(saveDir, [img_names(im_id).name(1:end-4) '_saturation.png']))
%     outlier_map = perform_robustfit(double(im_hsv(:,:,3)), mask);
%     imwrite(outlier_map, fullfile(saveDir, [img_names(im_id).name(1:end-4) '_value.png']))
    
    nR = 2; nC = 3;
    figure(1); subplot(nR, nC, 1); imshow(im);
    subplot(nR, nC, 2); imagesc(im_gray); axis image; colormap gray; title('gray')
    im_c = imread(fullfile(saveDir, [img_names(im_id).name(1:end-4) '_gray.png']));
    subplot(nR, nC, 3); imagesc(im_c); axis image; title('from gray')
    im_c = imread(fullfile(saveDir, [img_names(im_id).name(1:end-4) '_hue.png']));
    subplot(nR, nC, 4); imagesc(im_c); axis image; title('from hue')
    im_c = imread(fullfile(saveDir, [img_names(im_id).name(1:end-4) '_saturation.png']));
    subplot(nR, nC, 5); imagesc(im_c); axis image; title('from saturation')
    im_c = imread(fullfile(saveDir, [img_names(im_id).name(1:end-4) '_value.png']));
    subplot(nR, nC, 6); imagesc(im_c); axis image; title('from value')
    print(1, '-dpdf', fullfile(saveDir, img_names(im_id).name(1:end-4)));
    
end
end

function outlier_map = perform_robustfit(im_c, mask)
    [imH, imW, ~] = size(im_c);
    [C, R] = meshgrid(0:(imW-1), 0:(imH-1));

    Y = R(mask)/(imH-1) - 0.5;
    X = C(mask)/(imW-1) - 0.5;

    A = [X.^2, Y.^2, X.*Y, X, Y];
    %             A = [(X.^2 + Y.^2)];
    [b,stats] = robustfit(A,im_c(mask));
    
% %     close all;
%     figure(1); plot(sort(stats.resid), 'r');
%     hold on; line([1, imH*imW], 2*[stats.s, stats.s]); 
%     line([1, imH*imW], -2*[stats.s, stats.s]); 
%     hold off;
    
    outlier_map = zeros([imH, imW]);            
    outlier_map(mask) = abs(stats.resid) > 3*stats.s;

%     nR = 2; nC = 3;
%     figure(2); subplot(nR, nC, 1); imshow(im);
%     subplot(nR, nC, 2); imagesc(im_c); axis image; colormap gray;
%     subplot(nR, nC, 3); imagesc(im_gray); axis image; 
%     subplot(nR, nC, 4); imagesc(outlier_map); axis image;

    CC = bwconncomp(outlier_map);


    nCC = CC.NumObjects;
    ccSzs = zeros(1, nCC);
    for i=1:nCC
        ccSzs(i) = length(CC.PixelIdxList{i});
    end

    threshold = 10;
    for i=1:nCC
        if ccSzs(i) < threshold
            outlier_map(CC.PixelIdxList{i}) = 0;
        end
    end
    % convert 1 to 255, for visible display
    outlier_map(outlier_map==1) = 255;
    outlier_map = uint8(outlier_map);
end