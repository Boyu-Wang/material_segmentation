classdef process_data_jan
    
    methods(Static)
        % test different approaches for segmentation
        function main()
            bk = imread('../data/data_jan2019/EXPT3PressTime/EXPT3_FieldRef_50X.tif');
            bk_gray = rgb2gray(bk); bk_hsv = rgb2hsv(bk);
            [outlier_map_bk, ~, bk_region_loc, ~, ~] = process_data_jan.perform_robustfit(double(bk_gray), 3);
            im = imread('../data/data_jan2019/EXPT3PressTime/Long/tile_x003_y015.tif');
            im_gray = rgb2gray(im); im_hsv = rgb2hsv(im);
%             imd = double(im) - double(bk);
            imd = im - bk;
            imd_gray = rgb2gray(imd); imd_hsv = rgb2hsv(imd);
%             imd_gray = double(im_gray) - double(bk_gray);
            [outlier_map, region_labelmap, im_region_loc, ~, ~] = process_data_jan.perform_robustfit(double(im_gray), 3);
            [outlier_map_diff, ~, ~, ~, ~] = process_data_jan.perform_robustfit(double(imd_gray), 3);
            
            % remove regions from background image
            D = pdist2(im_region_loc, bk_region_loc);
            D_min = min(D, [], 2);
            to_remove = find(D_min < 5);
            outlier_map_rmv = zeros(size(region_labelmap));
            for i = 1: length(D_min)
                if ~ismember(i, to_remove)
                    outlier_map_rmv(region_labelmap==i) = 255;
                end
            end
            figure(1);
            subplot(3,3,1);imagesc(bk); title('background image');
            subplot(3,3,2);imagesc(im); title('ori image');
            subplot(3,3,3); imagesc(imd_gray); title('diff image');
            subplot(3,3,4); imagesc(outlier_map_bk); title('robust fit on bk')
            subplot(3,3,5); imagesc(outlier_map); title('robust fit on ori')
            subplot(3,3,6); imagesc(outlier_map_diff); title('robust fit on diff')
            subplot(3,3,7); imagesc(outlier_map - outlier_map_bk); title('robust fit ori - robust fit bk')
            subplot(3,3,8); imagesc(imd_gray(:,:,1)>5); title('diff threshold')
            subplot(3,3,9); imagesc(outlier_map_rmv); title('removed regions')
            
            B = bwboundaries(outlier_map_rmv, 'noholes');
            figure(2);
            imshow(im); title('overlay');
            hold on;
            for k = 1:length(B)
               boundary = B{k};
               plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
            end
        end
        
        % only one channel is not enough for some images, use gray, h, s
        % channel and combine them together
        function main2()
            bk = imread('../data/data_jan2019/EXPT3PressTime/EXPT3_FieldRef_50X.tif');
            bk_gray = rgb2gray(bk); bk_hsv = rgb2hsv(bk); mask_bk = bk_gray>0;
            [outlier_map_bk, ~, bk_region_loc, ~, ~] = process_data_jan.perform_robustfit(double(bk_gray), mask_bk, 3);
            im = imread('../data/data_jan2019/EXPT3PressTime/Long/tile_x003_y015.tif');
            im_gray = rgb2gray(im); im_hsv = rgb2hsv(im); mask = im_gray>0;
            imd = im - bk;
            imd_gray = rgb2gray(imd); imd_hsv = rgb2hsv(imd);
            [outlier_map_h, region_labelmap_h, im_region_loc_h, ~, ~] = process_data_jan.perform_robustfit(double(im_hsv(:,:,1)), mask, 20);
            [outlier_map_gray, region_labelmap_gray, im_region_loc_gray, ~, ~] = process_data_jan.perform_robustfit(double(im_gray(:,:,1)), mask, 3);
            [outlier_map_s, region_labelmap_s, im_region_loc_s, ~, ~] = process_data_jan.perform_robustfit(double(im_hsv(:,:,2)), mask, 15);
            
            % remove regions from background image
            D = pdist2(im_region_loc_h, bk_region_loc);
            D_min = min(D, [], 2);
            to_remove = find(D_min < 5);
            outlier_map_rmv_h = zeros(size(region_labelmap_h));
            for i = 1: length(D_min)
                if ~ismember(i, to_remove)
                    outlier_map_rmv_h(region_labelmap_h==i) = 255;
                end
            end
            D = pdist2(im_region_loc_gray, bk_region_loc);
            D_min = min(D, [], 2);
            to_remove = find(D_min < 5);
            outlier_map_rmv_gray = zeros(size(region_labelmap_gray));
            for i = 1: length(D_min)
                if ~ismember(i, to_remove)
                    outlier_map_rmv_gray(region_labelmap_gray==i) = 255;
                end
            end
            D = pdist2(im_region_loc_s, bk_region_loc);
            D_min = min(D, [], 2);
            to_remove = find(D_min < 5);
            outlier_map_rmv_s = zeros(size(region_labelmap_s));
            for i = 1: length(D_min)
                if ~ismember(i, to_remove)
                    outlier_map_rmv_s(region_labelmap_s==i) = 255;
                end
            end
            
            % at least two channels needs to be identified as flake
            or_map = outlier_map_rmv_h + outlier_map_rmv_gray + outlier_map_rmv_s;
            or_map(or_map< 255*2) = 0;
            or_map(or_map>=255*2) = 255;
            figure(1);
            subplot(3,3,1);imagesc(bk); title('background image');
            subplot(3,3,2);imagesc(im); title('ori image');
            subplot(3,3,3); imagesc(imd_gray); title('diff image');
            subplot(3,3,4); imagesc(outlier_map_bk); title('robust fit on bk')
            subplot(3,3,5); imagesc(outlier_map_h); title('robust fit on gray')
            subplot(3,3,6); imagesc(outlier_map_rmv_h); title('H')
            subplot(3,3,7); imagesc(outlier_map_gray); title('gray')
            subplot(3,3,8); imagesc(outlier_map_s); title('S')
            subplot(3,3,9); imagesc(or_map); title('or')
            B = bwboundaries(or_map, 'noholes');
            figure(2);
            imagesc(im); title('overlay');
            hold on;
            for k = 1:length(B)
               boundary = B{k};
               plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
            end

        end
        
        
        % region_meanlocs: [num_regions, 2]
        function [outlier_map, region_labelmap, region_meanlocs, region_size, num_regions] = perform_robustfit(im_c, mask, std_thre)
            [imH, imW, ~] = size(im_c);
            [C, R] = meshgrid(0:(imW-1), 0:(imH-1));

            Y = R(mask)/(imH-1) - 0.5;
            X = C(mask)/(imW-1) - 0.5;

            A = [X.^2, Y.^2, X.*Y, X, Y];
            %             A = [(X.^2 + Y.^2)];
            [b,stats] = robustfit(A,im_c(mask));


            outlier_map = zeros([imH, imW]);            
            outlier_map(mask) = abs(stats.resid) > std_thre*stats.s;
            stats.s

            CC = bwconncomp(outlier_map);

            nCC = CC.NumObjects;
            ccSzs = zeros(1, nCC);
            for i=1:nCC
                ccSzs(i) = length(CC.PixelIdxList{i});
            end
            
            threshold = 20;
            region_labelmap = zeros([imH, imW]);
            num_regions = 0;
            [~, region_idxs] = sort(ccSzs, 'descend');
            for i=1:nCC
                if ccSzs(i) < threshold
                    outlier_map(CC.PixelIdxList{i}) = 0;
                end
            end
            % convert 1 to 255, for visible display
            outlier_map(outlier_map==1) = 255;
            outlier_map = uint8(outlier_map);
            
            region_meanlocs = [];
            region_size = [];
            for i=1:nCC
                r_idx = region_idxs(i);
                if ccSzs(r_idx) >= threshold
                    num_regions = num_regions + 1;
                    region_labelmap(CC.PixelIdxList{r_idx}) = num_regions;
                    [r_r, r_c] = ind2sub([imH, imW], CC.PixelIdxList{r_idx});
                    region_meanlocs = [region_meanlocs; mean(r_r), mean(r_c)];
                    region_size = [region_size; ccSzs(r_idx)];
                end
            end
            
            
        end
        
        
    end
end
