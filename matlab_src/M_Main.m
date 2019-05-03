% things to try:
%{
1. background seems to be flat in h channel, background seems to fit well
in s/v channel ( can get rid of focal lense noise which exist in gray
image). However, flake is not consistent in s channel. Perform cluster in
different channels, and robust fit in different channels, and combine them
2. combine with edge detector.
%}
classdef M_Main
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 07-Oct-2018
% Last modified: 07-Oct-2018
    
    methods (Static)
        function main()
            dataDir = '../data/2D_materials/Graphene and Graphite/';
%             im = imread(sprintf('%s/01172018 g chip 1-000009.jpg', dataDir));
%             im = imread(sprintf('%s/07272018 G chip 7-000005.jpg', dataDir));
%             im = imread('/Users/BoyuWang/Downloads/Shared With Boyu/bilayer.jpg');
            bk = imread('~/Downloads/EXPT 2 Transfer Pressure/EXPT2_FieldRef_50X.tif');
            im = imread('~/Downloads/EXPT 2 Transfer Pressure/High Pressure/tile_x001_y001.tif');
            im = im - bk;
%             im = imresize(im, 0.25);
            im_hsv = rgb2hsv(im);
            im_gray = rgb2gray(im);
            im_c = double(im_gray(:,:,1));
%             im_c = double(im_hsv(:,:,1));
            

%             mask = im_gray(:) > 20; % pixels which are not completely black. The mask is to avoid black border.
            mask = im_gray(:) > 0;
            
            [imH, imW, ~] = size(im);
            [C, R] = meshgrid(0:(imW-1), 0:(imH-1));
            
            Y = R(mask)/(imH-1) - 0.5;
            X = C(mask)/(imW-1) - 0.5;
            
%             A = [X.^2, Y.^2, X.*Y, X, Y];
            A = ones(size(X));
%             A = [(X.^2 + Y.^2)];
%             A = [X, Y];
            [b,stats] = robustfit(A,im_c(mask));
            
            figure; plot(sort(stats.resid), 'r');
            hold on; line([1, imH*imW], 2*[stats.s, stats.s]); 
            line([1, imH*imW], -2*[stats.s, stats.s]); 
            
            outlier_map = zeros([imH, imW]);            
            outlier_map(mask) = abs(stats.resid) > 3*stats.s;
            
            nR = 2; nC = 3;
            figure; subplot(nR, nC, 1); imshow(im);
            subplot(nR, nC, 2); imagesc(im_c); axis image; colormap gray;
            subplot(nR, nC, 3); imagesc(im_gray); axis image; 
            subplot(nR, nC, 4); imagesc(outlier_map); axis image;
            
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
            subplot(nR, nC, 5); imagesc(outlier_map); axis image;
        end
    end
end

