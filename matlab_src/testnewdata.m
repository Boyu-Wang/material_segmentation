classdef testnewdata
    
    methods(Static)
        function main()
            % load the whole large image, and divide it and save them into multiple
            % patches
%             whole_im = testnewdata.load_data();
%             testnewdata.generate_patches(whole_im);
            
            patch_dir = '../data/data_20x';
            result_dir = '../results/data_20x_clustering';
            if ~exist(result_dir, 'dir')
                system(['mkdir ', result_dir]);
            end
            img_names = dir([patch_dir, '/patch*.png']);
            num_imgs = length(img_names);
%             for im_id = 1: 5
            all_region_feas = [];
            all_num_regions = zeros([1, num_imgs]);
            all_region_names = {};
            all_region_ids = [];
            %{
            for im_id = 1: num_imgs
                im = imread(sprintf('%s/%s', patch_dir, img_names(im_id).name));
%                 im = imread(sprintf('%s/patch-1-8.png', patch_dir));
%                  im = imread('/Users/BoyuWang/Downloads/Shared With Boyu/bilayer.jpg');
%                 im = imresize(im, 0.25);
                im_hsv = rgb2hsv(im);
                im_gray = rgb2gray(im);
                [res_robustfit, res_labelmap, region_fea] = testnewdata.perform_robustfit(double(im_gray(:,:,1)), im_gray, im, im_hsv);
                n_region_i = size(region_fea, 1);
                all_num_regions(1, im_id) = n_region_i;
                all_region_feas = [all_region_feas; region_fea];
                name_i = cell(n_region_i, 1);
                name_i(:) = {img_names(im_id).name};
                all_region_names = [all_region_names; name_i];
                all_region_ids = [all_region_ids; (1:n_region_i)'];
                save(fullfile(result_dir, sprintf('regionmap_%s.mat', img_names(im_id).name(1:end-4))), 'res_labelmap');
%                 nR = 2; nC = 2;
%                 figure(1); subplot(nR, nC, 1); imshow(im);
%                 subplot(nR, nC, 2); imagesc(im_gray); axis image; title('gray')
%                 subplot(nR, nC, 3); imagesc(res_robustfit); axis image; title('robustfit from gray')
%                 figure(100);
%                 res_watershed = testnewdata.perform_watershed(im_gray(:,:,1));
%                 print(100, '-dpdf', fullfile(result_dir, [img_names(im_id).name(1:end-4), '_watershed.pdf']));
%                 close 100;
%                 subplot(nR, nC, 4); imagesc(res_watershed); axis image; title('watershed from gray')
%                 print(1, '-dpdf', fullfile(result_dir, [img_names(im_id).name(1:end-4), '.pdf']));
%                 close all
            end
            save(fullfile(result_dir, 'all_region_fea.mat'), 'all_region_feas', 'all_num_regions', 'all_region_names', 'all_region_ids');            
            %}
            % perform clustering
            load(fullfile(result_dir, 'all_region_fea.mat'));
            num_cluster = 100;
            [region_assignment, region_cluster] = kmeans(all_region_feas, num_cluster);
            num_regions = size(all_region_feas, 1);
            region_cnt = zeros([1, num_regions]);
            for c_i=1:num_cluster
                num_region_ci = sum(region_assignment==c_i);
                region_ids = find(region_assignment==c_i);
                num_rows = 5;
%                 num_fig = ceil(num_region_ci/num_rows/num_rows);
                fig_id = 1;
                for im_i=1:num_region_ci
                    if mod(im_i, num_rows*num_rows) == 1
                        figure(1)
                    end
                    % locate the region
                    r_i = region_ids(im_i);
                    region_name_r = all_region_names{r_i};
                    region_id_r = all_region_ids(r_i);
                    res_labmap = load(fullfile(result_dir, sprintf('regionmap_%s.mat', region_name_r(1:end-4))));
                    res_labmap = res_labmap.res_labelmap;
                    [r, c] = find(res_labmap == region_id_r);
                    % read the image
                    im = imread(sprintf('%s/%s', patch_dir, region_name_r));
    %                 im = imresize(im, 0.25);
                    r_min = min(r); r_max = max(r);
                    c_min = min(c); c_max = max(c);
                    region = im(max(1, r_min-int16(0.1*(r_max-r_min))):min(size(im,1),r_max+int16(0.1*(r_max-r_min))), max(1, c_min-int16(0.1*(c_max-c_min))):min(size(im,2),c_max+int16(0.1*(c_max-c_min))),:);
                    
                    if mod(im_i, num_rows*num_rows) == 0
                        subplot(num_rows, num_rows, num_rows*num_rows)
                    else
                        subplot(num_rows, num_rows, mod(im_i, num_rows*num_rows))
                    end
                    imshow(region)
                    if mod(im_i, num_rows*num_rows) == 0 || im_i == num_region_ci
                        print(1, '-djpeg', fullfile(result_dir, sprintf('cluster-%d-%d.jpeg', c_i, fig_id)));
                        fig_id = fig_id + 1;
                        close(1)
                    end
                end
%                 close(1);
            end
%             % display each cluster
%             for r_i=1:num_regions
%                 % locate the regions
%                 region_name_r = all_region_names{r_i};
%                 region_id_r = all_region_ids(r_i);
%                 res_labmap = load(fullfile(result_dir, sprintf('regionmap_%s.mat', region_name_r(1:end-4))));
%                 res_labmap = res_labmap.res_labelmap;
%                 [r, c] = find(res_labmap == region_id_r);
%                 % read the image
%                 im = imread(sprintf('%s/%s', patch_dir, region_name_r));
% %                 im = imresize(im, 0.25);
%                 region = im(min(r):max(r), min(c):max(c),:);
%                 if ~isnan(region_assignment(r_i))
%                     region_cnt(region_assignment(r_i)) = region_cnt(region_assignment(r_i)) + 1;
%     %                 figure(1)
%                     imwrite(region, fullfile(result_dir, sprintf('cluster-%d_%d.png', region_assignment(r_i), region_cnt(region_assignment(r_i)))));
%                 end
%             end
            
        end

        function whole_im = load_data()
            whole_im = imread('../data/data_20x/Flakes_20X_2.tif');
            % take the upper part due to blurry of lower part
            whole_im = whole_im(1:6000, 1501:25000,:);
        end
        
        function generate_patches(whole_im)
            [whole_h, whole_w, ~] = size(whole_im);
            stride_h = 300;
            stride_w = 500;
            patch_h = 900;
            patch_w = 1200;
            patch_dir = '../data/data_20x/';
            system(['mkdir ' patch_dir]);
            start_idx_h = 1: stride_h: whole_h-patch_h+1;
            start_idx_w = 1: stride_w: whole_w-patch_w+1;
            for h_idx = 1: length(start_idx_h)
                parfor w_idx = 1: length(start_idx_h)
                    im_patch = whole_im(start_idx_h(h_idx): start_idx_h(h_idx)+patch_h-1, start_idx_w(w_idx): start_idx_w(w_idx)+patch_w-1,:);
                    imwrite(im_patch, fullfile(patch_dir, sprintf('patch-%d-%d.png', h_idx, w_idx)));
                end
            end

        end
        
        
        function [outlier_map, region_labelmap, region_fea] = perform_robustfit(im_c, im_gray, im_color, im_hsv)
            mask = im_gray(:) > 20; % pixels which are not completely black. The mask is to avoid black border.
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
            
            threshold = 10*4*4;
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
            
            for i=1:nCC
                r_idx = region_idxs(i);
                if ccSzs(r_idx) >= threshold
                    num_regions = num_regions + 1;
                    region_labelmap(CC.PixelIdxList{r_idx}) = num_regions;
                end
            end
            % get features for each region
            region_fea = zeros(num_regions, 7, 3);
            for i=1:num_regions
                % compute region avg, std, relative std for each channel
                for c=1:7
                    if c == 4
                        channel = im_gray;
                    elseif c < 4
                        channel = im_color(:,:,c);
                    else
                        channel = im_hsv(:,:,c-4);
                    end
                    region_val = double(channel(region_labelmap==i));
                    r_mean = mean(region_val);
                    r_std = std(region_val);
                    r_rvar = r_std / (r_mean + 1e-8);
                    region_fea(i, c, :) = [r_mean, r_std, r_rvar];
                end
            end
            region_fea = reshape(region_fea, [num_regions, 7*3]);
        end
        
        function L = perform_watershed(I)
%             figure(100);
            nr = 3;
            nc = 3;
            img_cnt = 1;
            gmag = imgradient(I);
%             figure(2)
            subplot(nr,nc, img_cnt); imshow(I); title('Gray Image'); img_cnt = img_cnt + 1;
            subplot(nr,nc, img_cnt); imshow(gmag,[]); title('Gradient Magnitude'); img_cnt = img_cnt + 1;
            L = watershed(gmag);
            Lrgb = label2rgb(L);
%             subplot(nr,nc, img_cnt); imshow(Lrgb); title('Watershed Transform of Gradient Magnitude'); img_cnt = img_cnt + 1;
            
            se = strel('disk',3);
%             Io = imopen(I,se);
%             subplot(nr,nc, 4); imshow(Io); title('Opening')
            
            Ie = imerode(I,se);
            Iobr = imreconstruct(Ie,I);
            subplot(nr,nc, img_cnt); imshow(Iobr); title('Opening-by-Reconstruction'); img_cnt = img_cnt + 1;
            
%             Ioc = imclose(Io,se);
%             subplot(nr,nc, 6); imshow(Ioc); title('Opening-Closing')
            
            Iobrd = imdilate(Iobr,se);
            Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
            Iobrcbr = imcomplement(Iobrcbr);
            subplot(nr,nc, img_cnt); imshow(Iobrcbr); title('Opening-Closing by Reconstruction'); img_cnt = img_cnt + 1;
            
%             fgm = imregionalmax(Iobrcbr);
            fgm = imbinarize(Iobrcbr, 'adaptive');
%             fgm = Iobrcbr < 85;
%             subplot(nr,nc, img_cnt); imshow(fgm); title('Regional Maxima of Opening-Closing by Reconstruction'); img_cnt = img_cnt + 1;
%             I2 = labeloverlay(I,fgm);
%             subplot(nr,nc, 9); imshow(I2); title('Regional Maxima Superimposed on Original Image')
            se2 = strel(ones(3,3));
            fgm2 = imclose(fgm,se2);
            fgm3 = imerode(fgm2,se2);
%             fgm4 = bwareaopen(fgm3,10);
            fgm4 = fgm;
%             I3 = labeloverlay(I,fgm4);
%             subplot(nr,nc, img_cnt); imshow(I3); title('Modified'); img_cnt = img_cnt + 1;
            subplot(nr,nc, img_cnt); imagesc(fgm4); title('Modified'); img_cnt = img_cnt + 1;
%             bw = imbinarize(Iobrcbr);
            bw = fgm;
%             level = graythresh(I);
%             bw = im2bw(I, level);
            subplot(nr,nc, img_cnt); imshow(bw); title('Thresholded'); img_cnt = img_cnt + 1;
            
            D = bwdist(bw);
            DL = watershed(D);
            bgm = DL == 0;
%             subplot(nr,nc, img_cnt); imshow(bgm); title('Watershed Ridge Lines)'); img_cnt = img_cnt + 1;
            
            gmag2 = imimposemin(gmag, bgm | fgm4);
            L = watershed(gmag2);
            % remove objects that are too big (caused by lines in the
            % strideing)
            unique_labels = unique(L);
            occurs = histc(L(:), unique_labels);
            bk_labels = unique_labels(occurs > 10000);
            if length(bk_labels) > 1
                L(ismember(L, bk_labels)) = bk_labels(1);
            end
            % remove the boundaries around the object
            L(L==0) = bk_labels(1);
%             [~, sort_label] = sort(occurs, 'descend');
%             L2 = zeros(size(L));
%             for l_i = 1:length(unique_labels)
%                 L2(L==(sort_label(l_i)-1)) = l_i - 1;
%             end

            labels = imdilate(L==0,ones(3,3)) + 2*bgm + 3*fgm4;
%             I4 = labeloverlay(I,labels);
            Lrgb = label2rgb(L,'jet','w','shuffle');
%             subplot(nr,nc, img_cnt); imshow(I4); title('Markers & Boundaries'); img_cnt = img_cnt + 1;
            subplot(nr,nc, img_cnt); imagesc(labels); title('Markers & Boundaries'); img_cnt = img_cnt + 1;
            subplot(nr,nc, img_cnt); imshow(Lrgb); title('Watershed Label Matrix'); img_cnt = img_cnt + 1;
        end
    end
end
