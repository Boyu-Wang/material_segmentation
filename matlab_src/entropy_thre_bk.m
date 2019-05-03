% use entropy thresh to filter out trash area
function entropy_thre(startim_id, endim_id)
run('/home/boyu/env/matlabconfig.m')
dataDir = '../data/2D_materials/Graphene and Graphite';
saveDir = '../results/bk_seg_robustfit';
img_names = dir([dataDir, '/0*.jpg']);
num_imgs = length(img_names);
for im_id = startim_id: endim_id
    ml_progressBar(im_id, endim_id);
%     im = imread(sprintf('%s/01172018 g chip 1-000006.jpg', dataDir));
    im = imread(sprintf('%s/%s', dataDir, img_names(im_id).name));
    im = imresize(im, 0.25);
    im_hsv = rgb2hsv(im);
    im_gray = rgb2gray(im);
%     im_c = double(im_gray(:,:,1));

    mask = im_gray(:) <= 20; % pixels which are not completely black. The mask is to avoid black border.
    hv_edge_low = 5;
    hv_edge_high = 30;
    hv_edge_alpha = 1;
    hv_entropy_min = 0;
    hv_entropy_max = 10;

    
    % Particle filter based on entropy thresholdings
%     Rgb1ToGray(ho_Image, &ho_ImageGrey);
    edge_im = edge(im_gray(:,:,1), 'canny', [hv_edge_low/255, hv_edge_high/255], hv_edge_alpha);
    edge_im(mask) = 0;
%     EdgesImage(ho_ImageGrey, &ho_ImaAmp, &ho_ImaDir, "canny",
%              hv_edge_alpha, "nms", hv_edge_low, hv_edge_high);
     
%     ClosingCircle(ho_ImaAmp, &ho_ImaAmp, 10);
%     DilationCircle(ho_ImaAmp, &ho_ImaAmp, 2);
%     FillUp(ho_ImaAmp, &ho_ImaAmp);
%     ErosionCircle(ho_ImaAmp, &ho_ImaAmp, 5);
    se = strel('disk',5);
    closed_im = imclose(edge_im,se);
%     closed_im = imfill(closed_im, 'holes');
    
%     Connection(ho_ImaAmp, &ho_ImaAmp);
    CC = bwconncomp(closed_im);
    nCC = CC.NumObjects;
    ccSzs = zeros(1, nCC);
    for i=1:nCC
        ccSzs(i) = length(CC.PixelIdxList{i});
    end
    
    entropy_im = closed_im;
    threshold = 10;
    for i=1:nCC
        if ccSzs(i) < threshold
            entropy_im(CC.PixelIdxList{i}) = 0;
        end
        region_entropy = entropy(im_gray(CC.PixelIdxList{i}));
        if region_entropy <= hv_entropy_min || region_entropy >= hv_entropy_max
            entropy_im(CC.PixelIdxList{i}) = 0;
        end
    end
%     BitRshift(ho_ImageGrey, &ho_ImageGrey_conv, 4);
%     ConvertImageType(ho_ImageGrey_conv, &ho_ImageGrey_8, "byte");

%     // Entropy filter
%     SelectGray(ho_ImaAmp, ho_ImageGrey_8, &ho_RegionS, "entropy",
%              "and", hv_entropy_min, hv_entropy_max);
    
% 
%     DilationCircle(ho_RegionS, &ho_RegionS, 2);
%     Union1(ho_RegionS, &ho_RegionS);
%     OpeningCircle(ho_RegionS, &ho_RegionS, 3);
%     FillUp(ho_RegionS, &ho_RegionS);
%     ErosionCircle(ho_RegionS, &ho_RegionS, 2);

%     // Thresholding and particle filter
%     Intersection(ho_RegionI, ho_RegionS, &ho_RegionIII);
    entropy_im = uint8(entropy_im);
    entropy_im(entropy_im==1) = 255;
    
    edge_im = uint8(edge_im);
    edge_im(edge_im==1) = 255;
    imwrite(edge_im, fullfile(saveDir, [img_names(im_id).name(1:end-4) '_edge.png']))
    imwrite(entropy_im, fullfile(saveDir, [img_names(im_id).name(1:end-4) '_entropy.png']))
    
    nR = 2; nC = 3;
    figure(1); subplot(nR, nC, 1); imshow(im);
    subplot(nR, nC, 2); imagesc(im_gray); axis image; colormap gray; title('gray')
    subplot(nR, nC, 3); imagesc(edge_im); axis image; title('edges')
    subplot(nR, nC, 4); imagesc(closed_im); axis image; title('morphology')
    subplot(nR, nC, 5); imagesc(entropy_im); axis image; title('entropy')
    print(1, '-dpdf', fullfile(saveDir, [img_names(im_id).name(1:end-4), '_entropythre.pdf']));
end
end
