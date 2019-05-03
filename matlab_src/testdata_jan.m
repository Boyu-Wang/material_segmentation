classdef testdata_jan
    
    methods(Static)
        % test different approaches for segmentation
        function test()
            bk = imread('../data/data_jan2019/EXPT2TransferPressure/EXPT2_FieldRef_50X.tif');
            bk_gray = rgb2gray(bk); bk_hsv = rgb2hsv(bk);
            [outlier_map_bk, ~, bk_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(bk_gray(:,:,1)), bk_gray);
%             im = imread('../data/data_jan2019/EXPT2TransferPressure/HighPressure/tile_x001_y001.tif');
%             im = imread('../data/data_jan2019/EXPT2TransferPressure/HighPressure/tile_x002_y011.tif');
%             im = imread('../data/data_jan2019/YoungJaeShinSamples/5/tile_x001_y002.tif');
            im = imread('../data/data_jan2019/EXPT3 PressTime/Long/tile_x001_y014.tif');
            im_gray = rgb2gray(im); im_hsv = rgb2hsv(im);
%             imd = double(im) - double(bk);
            imd = im - bk;
            imd_gray = rgb2gray(imd); imd_hsv = rgb2hsv(imd);
%             imd_gray = double(im_gray) - double(bk_gray);
            [outlier_map, region_labelmap, im_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(im_gray(:,:,1)), im_gray);
%             [outlier_map_diff, ~] = testdata_jan.perform_robustfit(double(imd_gray(:,:,1)), imd_gray, imd, imd_hsv);
            [outlier_map_diff, ~, diff_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(imd_gray(:,:,1)), imd_gray);
%             outlier_map_watershed = testdata_jan.perform_watershed(double(imd_gray(:,:,1)));
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
            subplot(3,3,9); imagesc(outlier_map_rmv); title('remove regions')
            figure(2);
            imshow(im); title('remove regions');
            B = bwboundaries(outlier_map_rmv, 'noholes');
            hold on;
            for k = 1:length(B)
               boundary = B{k};
               plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
            end
%             subplot(3,3,9); imagesc(outlier_map_watershed); title('watershed on diff')
        end
        
        function test2()
%             bk = imread('../data/data_jan2019/EXPT2TransferPressure/EXPT2_FieldRef_50X.tif');
            bk = imread('../data/data_jan2019/EXPT2TransferPressure/EXPT2_FieldRef_50X.tif');
            bk_gray = rgb2gray(bk); bk_hsv = rgb2hsv(bk);
            [outlier_map_bk, ~, bk_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(bk_gray(:,:,1)), bk_gray, 10);
%             im = imread('../data/data_jan2019/EXPT2TransferPressure/HighPressure/tile_x001_y001.tif');
%             im = imread('../data/data_jan2019/YoungJaeShinSamples/5/tile_x001_y002.tif');
            im = imread('../data/data_jan2019/EXPT3PressTime/Long/tile_x003_y015.tif');
            im_gray = rgb2gray(im); im_hsv = rgb2hsv(im);
%             imd = double(im) - double(bk);
            imd = im - bk;
            imd_gray = rgb2gray(imd); imd_hsv = rgb2hsv(imd);
%             imd_gray = double(im_gray) - double(bk_gray);
%             [outlier_map, res_map, region_labelmap, im_region_loc, ~, ~] = testdata_jan.perform_robustfit_multichannel(im_hsv, im_gray, 20);
            [outlier_map, region_labelmap, im_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(im_hsv(:,:,1)), im_gray, 20);
            [outlier_map_gray, region_labelmap_gray, im_region_loc_gray, ~, ~] = testdata_jan.perform_robustfit(double(im_gray(:,:,1)), im_gray, 10);
            [outlier_map_s, region_labelmap_s, im_region_loc_s, ~, ~] = testdata_jan.perform_robustfit(double(im_hsv(:,:,2)), im_gray, 10);
            D = pdist2(im_region_loc, bk_region_loc);
            D_min = min(D, [], 2);
            to_remove = find(D_min < 5);
            outlier_map_rmv = zeros(size(region_labelmap));
            for i = 1: length(D_min)
                if ~ismember(i, to_remove)
                    outlier_map_rmv(region_labelmap==i) = 255;
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
            or_map = outlier_map_rmv + outlier_map_rmv_gray + outlier_map_rmv_s;
            or_map(or_map< 255*2) = 0;
            or_map(or_map>=255*2) = 255;
            figure(1);
            subplot(3,3,1);imagesc(bk); title('background image');
            subplot(3,3,2);imagesc(im); title('ori image');
            subplot(3,3,3); imagesc(imd_gray); title('diff image');
            subplot(3,3,4); imagesc(outlier_map_bk); title('robust fit on bk')
            subplot(3,3,5); imagesc(outlier_map); title('robust fit on ori')
%             subplot(3,3,6); imagesc(outlier_map_diff); title('robust fit on diff')
%             subplot(3,3,7); imagesc(outlier_map - outlier_map_bk); title('robust fit ori - robust fit bk')
%             subplot(3,3,8); imagesc(imd_gray(:,:,1)>5); title('diff threshold')
            subplot(3,3,6); imagesc(outlier_map_rmv); title('H')
            subplot(3,3,7); imagesc(outlier_map_gray); title('gray')
            subplot(3,3,8); imagesc(outlier_map_s); title('S')
            subplot(3,3,9); imagesc(or_map); title('or')
            figure(2);
%             im21 = im(:,:,1);im22 = im(:,:,2);im23 = im(:,:,3);
%             im21(outlier_map_rmv==255)= 255;%im22(outlier_map_rmv==255)= 255;
%             im23(outlier_map_rmv==255)= 255;
%             im2 = cat(3, [im21, im22, im23]);
%             im2 = reshape(im2, [], size(im2,2)/3, 3);
            imshow(im_gray); title('remove regions');
            B = bwboundaries(outlier_map_rmv, 'noholes');
            hold on;
            for k = 1:length(B)
               boundary = B{k};
               plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
            end
%             subplot(3,3,9); imagesc(outlier_map_watershed); title('watershed on diff')
        
        end
        
        function test3()
%             bk = imread('../data/data_jan2019/EXPT2TransferPressure/EXPT2_FieldRef_50X.tif');
            bk = imread('../data/data_jan2019/EXPT2TransferPressure/EXPT2_FieldRef_50X.tif');
            bk_gray = rgb2gray(bk); bk_hsv = rgb2hsv(bk);
            [outlier_map_bk, ~, bk_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(bk_gray(:,:,1)), bk_gray, 10);
%             im = imread('../data/data_jan2019/EXPT2TransferPressure/HighPressure/tile_x001_y001.tif');
%             im = imread('../data/data_jan2019/YoungJaeShinSamples/5/tile_x001_y002.tif');
            im = imread('../data/data_jan2019/EXPT3PressTime/Long/tile_x003_y015.tif');
            im_gray = rgb2gray(im); im_hsv = rgb2hsv(im);
%             imd = double(im) - double(bk);
            imd = im - bk;
            imd_gray = rgb2gray(imd); imd_hsv = rgb2hsv(imd);
%             imd_gray = double(im_gray) - double(bk_gray);
            [outlier_map_sum, res_map, region_labelmap, im_region_loc_sum, ~, ~] = testdata_jan.perform_robustfit_multichannel(im_hsv, im_gray, 10);
%             [outlier_map, region_labelmap, im_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(im_hsv(:,:,1)), im_gray, 20);
%             [outlier_map_gray, region_labelmap_gray, im_region_loc_gray, ~, ~] = testdata_jan.perform_robustfit(double(im_gray(:,:,1)), im_gray, 10);
%             [outlier_map_s, region_labelmap_s, im_region_loc_s, ~, ~] = testdata_jan.perform_robustfit(double(im_hsv(:,:,2)), im_gray, 10);
            D = pdist2(im_region_loc_sum, bk_region_loc);
            D_min = min(D, [], 2);
            to_remove = find(D_min < 5);
            outlier_map_rmv = zeros(size(region_labelmap));
            for i = 1: length(D_min)
                if ~ismember(i, to_remove)
                    outlier_map_rmv(region_labelmap==i) = 255;
                end
            end
%             D = pdist2(im_region_loc_gray, bk_region_loc);
%             D_min = min(D, [], 2);
%             to_remove = find(D_min < 5);
%             outlier_map_rmv_gray = zeros(size(region_labelmap_gray));
%             for i = 1: length(D_min)
%                 if ~ismember(i, to_remove)
%                     outlier_map_rmv_gray(region_labelmap_gray==i) = 255;
%                 end
%             end
%             D = pdist2(im_region_loc_s, bk_region_loc);
%             D_min = min(D, [], 2);
%             to_remove = find(D_min < 5);
%             outlier_map_rmv_s = zeros(size(region_labelmap_s));
%             for i = 1: length(D_min)
%                 if ~ismember(i, to_remove)
%                     outlier_map_rmv_s(region_labelmap_s==i) = 255;
%                 end
%             end
%             or_map = outlier_map_rmv + outlier_map_rmv_gray + outlier_map_rmv_s;
%             or_map(or_map< 255*2) = 0;
%             or_map(or_map>=255*2) = 255;
            figure(1);
            subplot(2,3,1);imagesc(bk); title('background image');
            subplot(2,3,2);imagesc(im); title('ori image');
            subplot(2,3,3); imagesc(imd_gray); title('diff image');
            subplot(2,3,4); imagesc(res_map); title('residual map');
            subplot(2,3,5); imagesc(outlier_map_sum); title('result');
            subplot(2,3,6); imshow(im_gray); title('overlay');
            B = bwboundaries(outlier_map_sum, 'noholes');
            hold on;
            for k = 1:length(B)
               boundary = B{k};
               plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
            end
            hold off;
%             subplot(3,3,4); imagesc(outlier_map_bk); title('robust fit on bk')
%             subplot(3,3,5); imagesc(outlier_map_sum); title('robust fit on ori')
% %             subplot(3,3,6); imagesc(outlier_map_diff); title('robust fit on diff')
% %             subplot(3,3,7); imagesc(outlier_map - outlier_map_bk); title('robust fit ori - robust fit bk')
% %             subplot(3,3,8); imagesc(imd_gray(:,:,1)>5); title('diff threshold')
%             subplot(3,3,6); imagesc(outlier_map_rmv); title('H')
%             subplot(3,3,7); imagesc(outlier_map_gray); title('gray')
%             subplot(3,3,8); imagesc(outlier_map_s); title('S')
%             subplot(3,3,9); imagesc(or_map); title('or')
%             figure(2);
% %             im21 = im(:,:,1);im22 = im(:,:,2);im23 = im(:,:,3);
% %             im21(outlier_map_rmv==255)= 255;%im22(outlier_map_rmv==255)= 255;
% %             im23(outlier_map_rmv==255)= 255;
% %             im2 = cat(3, [im21, im22, im23]);
% %             im2 = reshape(im2, [], size(im2,2)/3, 3);
%             imshow(im_gray); title('remove regions');
%             B = bwboundaries(outlier_map_rmv, 'noholes');
%             hold on;
%             for k = 1:length(B)
%                boundary = B{k};
%                plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
%             end
%             subplot(3,3,9); imagesc(outlier_map_watershed); title('watershed on diff')
        
        end
        
        function main()
            exp_path = '../data/data_jan2019';
            result_path = '../results/data_jan2019';
            if ~exist(result_path, 'dir')
                system(['mkdir ', result_path]);
            end
            exp_names = dir(exp_path);
            exp_names = exp_names(3:end);
            for d = 1: length(exp_names)
                exp_full_path = fullfile(exp_path, exp_names(d).name);
                result_full_path = fullfile(result_path, exp_names(d).name);
                if ~exist(result_full_path, 'dir')
                    system(['mkdir ', result_full_path]);
                end
                subexp_names = dir(exp_full_path);
                subexp_names = subexp_names(3:end);
                bk_name = [];
                subdir_names = [];
                for j = 1: length(subexp_names)
                    if ~isempty(strfind(subexp_names(j).name, 'FieldRef_50X.tif'))
                        bk_name = subexp_names(j).name;
%                     elseif isempty(strfind(subexp_names(j).name, 'FieldRef_50X_withScale.tif'))
                    elseif exist(fullfile(exp_full_path, subexp_names(j).name), 'dir')
                        subdir_names = [subdir_names; subexp_names(j)];
                    end
                end
                % read bk file
                bk_img_name = fullfile(exp_full_path, bk_name);
                bk_img = imread(bk_img_name);
                % find regions in bk
                bk_gray = rgb2gray(bk_img); bk_hsv = rgb2hsv(bk_img);
                [~, ~, bk_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(bk_gray(:,:,1)), bk_gray);
                % process each subexp
                for s_i = 1: length(subdir_names)
                    subexp_full_path = fullfile(exp_full_path, subdir_names(s_i).name);
                    subresult_full_path = fullfile(result_full_path,  subdir_names(s_i).name);
                    if ~exist(subresult_full_path, 'dir')
                        system(['mkdir ', subresult_full_path]);
                    end
                    img_names = dir(subexp_full_path);
                    img_names = img_names(3:end);
                    parfor im_i = 1:length(img_names)
                        img_name = fullfile(subexp_full_path, img_names(im_i).name);
                        save_name = fullfile(subresult_full_path, img_names(im_i).name(1:end-4));
                        testdata_jan.process_one_img(img_name, bk_img_name, bk_region_loc, save_name)
                    end
                end
                
            end
                 
        end
        
        % sum the residual of robustfit on multiple channels
        function main2(d_s, d_e)
            exp_path = '../data/data_jan2019';
            result_path = '../results/data_jan2019_sum/mat';
%             result_fig_path = '../results/data_jan2019_sum/fig';
            result_fig_path = '../results/data_jan2019_sum/info_mat';
            if ~exist(result_path, 'dir')
                system(['mkdir ', result_path]);
            end
            if ~exist(result_fig_path, 'dir')
                system(['mkdir ', result_fig_path]);
            end
            exp_names = dir(exp_path);
            exp_names = exp_names(3:end);
%             parpool('local',24)
%             for d = 1: 1%length(exp_names)
            for d = d_s:d_e
                exp_full_path = fullfile(exp_path, exp_names(d).name);
                result_full_path = fullfile(result_path, exp_names(d).name);
                result_fig_full_path = fullfile(result_fig_path, exp_names(d).name);
                if ~exist(result_full_path, 'dir')
                    system(['mkdir ', result_full_path]);
                end
                if ~exist(result_fig_full_path, 'dir')
                    system(['mkdir ', result_fig_full_path]);
                end
                subexp_names = dir(exp_full_path);
                subexp_names = subexp_names(3:end);
                bk_name = [];
                subdir_names = [];
                for j = 1: length(subexp_names)
                    if ~isempty(strfind(subexp_names(j).name, 'FieldRef_50X.tif'))
                        bk_name = subexp_names(j).name;
%                     elseif isempty(strfind(subexp_names(j).name, 'FieldRef_50X_withScale.tif'))
                    elseif exist(fullfile(exp_full_path, subexp_names(j).name), 'dir')
                        subdir_names = [subdir_names; subexp_names(j)];
                    end
                end
                % read bk file
                bk_img_name = fullfile(exp_full_path, bk_name);
                bk_img = imread(bk_img_name);
                % find regions in bk
                bk_gray = rgb2gray(bk_img); bk_hsv = rgb2hsv(bk_img);
                [~, ~, bk_region_loc, ~, ~] = testdata_jan.perform_robustfit(double(bk_gray(:,:,1)), bk_gray, 10);
                % process each subexp
                for s_i = 1: length(subdir_names)
                    subexp_full_path = fullfile(exp_full_path, subdir_names(s_i).name);
                    subresult_full_path = fullfile(result_full_path,  subdir_names(s_i).name);
                    subresult_fig_full_path = fullfile(result_fig_full_path,  subdir_names(s_i).name);
                    if ~exist(subresult_full_path, 'dir')
                        system(['mkdir ', subresult_full_path]);
                    end
                    if ~exist(subresult_fig_full_path, 'dir')
                        system(['mkdir ', subresult_fig_full_path]);
                    end
                    img_names = dir(subexp_full_path);
                    img_names = img_names(3:end);
                    parfor im_i = 1:length(img_names)
                        img_name = fullfile(subexp_full_path, img_names(im_i).name);
                        save_name = fullfile(subresult_full_path, img_names(im_i).name(1:end-4));
                        fig_save_name = fullfile(subresult_fig_full_path, img_names(im_i).name(1:end-4));
%                         if ~exist([save_name, '.mat'], 'file') || ~exist([fig_save_name, '.png'], 'file')
                        if ~exist([fig_save_name, '.mat'], 'file')
%                             fprintf('process')
%                             save_name
%                             fig_save_name
%                             testdata_jan.process_one_img2(img_name, bk_img_name, bk_region_loc, save_name, fig_save_name, im_i)
                            testdata_jan.process_one_img2_getbw(img_name, bk_img_name, bk_region_loc, save_name, fig_save_name, im_i)
                        end
                    end
                end
                
            end
                 
        end
        
        
        % plot for each subexp
        function plot_flake()
            data_path = '../data/data_jan2019';
            result_path = '../results/data_jan2019';
            plt_result_path = '../results/data_jan2019_plotflake';
            if ~exist(plt_result_path, 'dir')
                system(['mkdir ', plt_result_path]);
            end
            exp_names = dir(result_path);
            exp_names = exp_names(3:end);
            subdata_full_paths = [];
            subexp_full_paths = [];
            subplt_full_paths = [];
            fig_ids = [];
            cnt = 0;
            for d = 1: length(exp_names)
                data_full_path = fullfile(data_path, exp_names(d).name);
                exp_full_path = fullfile(result_path, exp_names(d).name);
                plt_full_path = fullfile(plt_result_path, exp_names(d).name);
                if ~exist(plt_full_path, 'dir')
                    system(['mkdir ', plt_full_path]);
                end
                subexp_names = dir(exp_full_path);
                subexp_names = subexp_names(3:end);
                % process each subexp
                for s_i = 1: length(subexp_names)
                    subdata_full_path = fullfile(data_full_path, subexp_names(s_i).name);
                    subexp_full_path = fullfile(exp_full_path, subexp_names(s_i).name);
                    subplt_full_path = fullfile(plt_full_path, subexp_names(s_i).name);
                    if ~exist(subplt_full_path, 'dir')
                        system(['mkdir ', subplt_full_path]);
                    end
                    subdata_full_paths = [subdata_full_paths; {subdata_full_path}];
                    subexp_full_paths = [subexp_full_paths; {subexp_full_path}];
                    subplt_full_paths = [subplt_full_paths; {subplt_full_path}];
                    cnt = cnt + 1;
                    fig_ids = [fig_ids; cnt];
                end
            end
            for i = 1: cnt
                testdata_jan.plot_flake_helper(i, subexp_full_paths{i}, subdata_full_paths{i}, subplt_full_paths(i))
%                     img_names = dir(subexp_full_path);
%                     img_names = img_names(3:end);
%                     figure(1)
%                     for im_i = 1:length(img_names)
%                         img_name = fullfile(subexp_full_path, img_names(im_i).name);
%                         rslt = load(img_name);
%                         ori_img_name = fullfile(subdata_full_path, [img_names(im_i).name(1:end-4), '.tif']);
%                         ori_img = imread(ori_img_name);
%                         subplot(2,1,1); imagesc(ori_img);
%                         subplot(2,1,2); imagesc(rslt.new_region_labelmap);
%                         print(1, '-dpng', fullfile(subplt_full_path,  [img_names(im_i).name(1:end-4), '.png']));
%                     end
%                 end
                
            end
        end
        
        % plot each subdirectory
        function plot_flake_helper(fig_id, subexp_full_path, subdata_full_path, subplt_full_path)
            figure(fig_id)
            img_names = dir(subexp_full_path);
            img_names = img_names(3:end);
            for im_i = 1:length(img_names)
                img_name = fullfile(subexp_full_path, img_names(im_i).name);
                rslt = load(img_name);
                ori_img_name = fullfile(subdata_full_path, [img_names(im_i).name(1:end-4), '.tif']);
                ori_img = imread(ori_img_name);
                subplot(2,1,1); imagesc(ori_img);
                subplot(2,1,2); imagesc(rslt.new_region_labelmap);
                print(fig_id, '-dpng', fullfile(subplt_full_path,  [img_names(im_i).name(1:end-4), '.png']));
            end
        end
        
         % plot stats for each subexp
        function plot_stats()
            data_path = '../data/data_jan2019';
            result_path = '../results/data_jan2019';
            plt_result_path = '../results/data_jan2019_plot';
            if ~exist(plt_result_path, 'dir')
                system(['mkdir ', plt_result_path]);
            end
            exp_names = dir(result_path);
            exp_names = exp_names(3:end);
            for d = 1: length(exp_names)
                data_full_path = fullfile(data_path, exp_names(d).name);
                exp_full_path = fullfile(result_path, exp_names(d).name);
                plt_full_path = fullfile(plt_result_path, exp_names(d).name);
                if ~exist(plt_full_path, 'dir')
                    system(['mkdir ', plt_full_path]);
                end
                subexp_names = dir(exp_full_path);
                subexp_names = subexp_names(3:end);
                % process each subexp
                for s_i = 1: length(subexp_names)
                    subdata_full_path = fullfile(data_full_path, subexp_names(s_i).name);
                    subexp_full_path = fullfile(exp_full_path, subexp_names(s_i).name);
                    subplt_full_path = fullfile(plt_full_path, subexp_names(s_i).name);
                    if ~exist(subplt_full_path, 'dir')
                        system(['mkdir ', subplt_full_path]);
                    end
                    img_names = dir(subexp_full_path);
                    img_names = img_names(3:end);
                    save_name = [exp_names(d).name, subexp_names(s_i).name];
                    region_size = [];
                    region_contrast = [];
%                     cnt = 0;
%                     for im_i = 1:length(img_names)
%                         img_name = fullfile(subexp_full_path, img_names(im_i).name);
%                         rslt = load(img_name);
%                         region_size = [region_size; rslt.new_region_size];
%                         region_contrast = [region_contrast; rslt.region_feas];
%                         cnt = cnt + rslt.num_regions;
%                     end
%                     num_regions = cnt;
                    load(fullfile(plt_result_path, save_name));
                    cnt = num_regions / length(img_names);
                    fprintf('%s: %d \n', save_name, cnt);
%                     region_contrast_gray = region_contrast(:,1);
%                     region_contrast_v = region_contrast(:,2);
%                     save(fullfile(plt_result_path, save_name), 'num_regions', 'region_size', 'region_contrast_gray', 'region_contrast_v');
                    % plot figure
                    log_region_size = log10(region_size);
                    figure(1); hist(log_region_size,100)
%                     x=logspace(-2,6,100);
%                     figure(1); hist(region_size, x); set(gca,'XScale','log');  set(gca,'YScale','log');
                    print(1, '-dpng', fullfile(plt_result_path, sprintf('%s_size.png', save_name)));
%                     figure(2); hist(region_contrast(:,1) - min(region_contrast(:,1)) + 1, 100); set(gca,'XScale','log');
                    log_region_contrast_gray = log10(region_contrast_gray - min(region_contrast_gray) + 1);
                    figure(2); hist(log_region_contrast_gray, 100);
                    print(2, '-dpng', fullfile(plt_result_path, sprintf('%s_contrast_gray.png', save_name)));
%                     figure(3); hist(region_contrast(:,2) - min(region_contrast(:,2))  + 1, 100); set(gca,'XScale','log');
                    log_region_contrast_v = log10(region_contrast_v - min(region_contrast_v) + 1);
                    figure(3); hist(log_region_contrast_v, 100);
                    print(3, '-dpng', fullfile(plt_result_path, sprintf('%s_contrast_v.png', save_name)));
%                     figure(4); hist3([region_size, region_contrast(:,1)- min(region_contrast(:,1)) + 1], [100,100]); set(gca,'XScale','log');
                    figure(4); hist3([log_region_size, log_region_contrast_gray], [50,50]);
                    print(4, '-dpng', fullfile(plt_result_path, sprintf('%s_size_contrastgray.png', save_name)));
%                     figure(5); hist3([region_size, region_contrast(:,2) - min(region_contrast(:,2)) + 1], [100, 100]);set(gca,'XScale','log');
                    figure(5); hist3([log_region_size, log_region_contrast_v], [50,50]);
                    print(5, '-dpng', fullfile(plt_result_path, sprintf('%s_size_contrastv.png', save_name)));
                end
                
            end
        end
        
        
        function getflake_info()
            data_path = '../data/data_jan2019';
            result_path = '../results/data_jan2019';
            plt_result_path = '../results/data_jan2019_flakeinfo';
            if ~exist(plt_result_path, 'dir')
                system(['mkdir ', plt_result_path]);
            end
            exp_names = dir(result_path);
            exp_names = exp_names(3:end);
            for d = 1: length(exp_names)
                data_full_path = fullfile(data_path, exp_names(d).name);
                exp_full_path = fullfile(result_path, exp_names(d).name);
                plt_full_path = fullfile(plt_result_path, exp_names(d).name);
                if ~exist(plt_full_path, 'dir')
                    system(['mkdir ', plt_full_path]);
                end
                subexp_names = dir(exp_full_path);
                subexp_names = subexp_names(3:end);
                % process each subexp
                for s_i = 1: length(subexp_names)
                    subdata_full_path = fullfile(data_full_path, subexp_names(s_i).name);
                    subexp_full_path = fullfile(exp_full_path, subexp_names(s_i).name);
                    subplt_full_path = fullfile(plt_full_path, subexp_names(s_i).name);
                    if ~exist(subplt_full_path, 'dir')
                        system(['mkdir ', subplt_full_path]);
                    end
                    img_names = dir(subexp_full_path);
                    img_names = img_names(3:end);
                    save_name = [exp_names(d).name, subexp_names(s_i).name];
                    region_size = [];
                    region_contrast = [];
                    sample_names = {};
                    region_bbox = [];
                    region_center = [];
                    cnt = 0;
%                     for im_i = 1:length(img_names)
%                         img_name = fullfile(subexp_full_path, img_names(im_i).name);
%                         rslt = load(img_name);
%                         region_size = [region_size; rslt.new_region_size];
%                         region_contrast = [region_contrast; rslt.region_feas];
%                         name_i = cell(rslt.num_regions, 1);
%                         name_i(:) = {img_names(im_i).name};
%                         sample_names = [sample_names; name_i];
%                         for f_i = 1: rslt.num_regions
%                             % get center and bbox for each flake
%                             [r_r, r_c] = find(rslt.new_region_labelmap == f_i);
%                             region_center = [region_center; mean(r_r), mean(r_c)];
%                             region_bbox = [region_bbox; min(r_r), max(r_r), min(r_c), max(r_c)];
%                         end
%                         cnt = cnt + rslt.num_regions;
%                     end
%                     num_regions = cnt;
                    load(fullfile(plt_result_path, save_name));
                    cnt = num_regions / length(img_names);
                    fprintf('%s: %d \n', save_name, cnt);
                    region_contrast_gray = region_contrast(:,1);
                    region_contrast_v = region_contrast(:,2);
%                     save(fullfile(plt_result_path, save_name), 'num_regions', 'region_size', 'region_contrast_gray', 'region_contrast_v', 'region_center', 'region_bbox', 'sample_names');
                    % plot figure
                    log_region_size = log10(region_size);
                    figure(1); hist(log_region_size,100)
%                     x=logspace(-2,6,100);
%                     figure(1); hist(region_size, x); set(gca,'XScale','log');  set(gca,'YScale','log');
                    print(1, '-dpng', fullfile(plt_result_path, sprintf('%s_size.png', save_name)));
%                     figure(2); hist(region_contrast(:,1) - min(region_contrast(:,1)) + 1, 100); set(gca,'XScale','log');
                    log_region_contrast_gray = log10(region_contrast_gray - min(region_contrast_gray) + 1);
                    figure(2); hist(log_region_contrast_gray, 100);
                    print(2, '-dpng', fullfile(plt_result_path, sprintf('%s_contrast_gray.png', save_name)));
%                     figure(3); hist(region_contrast(:,2) - min(region_contrast(:,2))  + 1, 100); set(gca,'XScale','log');
                    log_region_contrast_v = log10(region_contrast_v - min(region_contrast_v) + 1);
                    figure(3); hist(log_region_contrast_v, 100);
                    print(3, '-dpng', fullfile(plt_result_path, sprintf('%s_contrast_v.png', save_name)));
%                     figure(4); hist3([region_size, region_contrast(:,1)- min(region_contrast(:,1)) + 1], [100,100]); set(gca,'XScale','log');
                    figure(4); hist3([log_region_size, log_region_contrast_gray], [50,50]);
                    print(4, '-dpng', fullfile(plt_result_path, sprintf('%s_size_contrastgray.png', save_name)));
%                     figure(5); hist3([region_size, region_contrast(:,2) - min(region_contrast(:,2)) + 1], [100, 100]);set(gca,'XScale','log');
                    figure(5); hist3([log_region_size, log_region_contrast_v], [50,50]);
                    print(5, '-dpng', fullfile(plt_result_path, sprintf('%s_size_contrastv.png', save_name)));
                end
            end
        end
        
        % use robust fit get flakes in the image, save each region, # of
        % flakes, contrast of each region
        function process_one_img(img_name, bk_img_name, bk_region_loc, save_name)
            bk_im = imread(bk_img_name);
            bk_gray = rgb2gray(bk_im); bk_hsv = rgb2hsv(bk_im); bk_v = bk_hsv(:,:,3);
            im = imread(img_name);
            im_gray = rgb2gray(im); im_hsv = rgb2hsv(im); im_v = im_hsv(:,:,3);
%             imd = im - bk_im;
%             imd_gray = rgb2gray(imd); imd_hsv = rgb2hsv(imd);
            [outlier_map_diff, region_labelmap, region_meanlocs, region_size, num_regions] = testdata_jan.perform_robustfit(double(im_gray(:,:,1)), im_gray);
            % remove bk ground regions
            D = pdist2(region_meanlocs, bk_region_loc);
            D_min = min(D, [], 2);
            to_remove = find(D_min < 5);
            new_region_labelmap = zeros(size(region_labelmap));
            new_region_size = [];
            region_feas = [];
            cnt = 0;
            for i = 1: num_regions
                if ~ismember(i, to_remove)
                    cnt = cnt + 1;
                    new_region_labelmap(region_labelmap==i) = cnt;
                    new_region_size = [new_region_size; region_size(i)];
                    % get feature for the region
                    tmp_fea = [mean(double(im_gray(region_labelmap==i)) - double(bk_gray(region_labelmap==i))), ...
                        mean(double(im_v(region_labelmap==i)) - double(bk_v(region_labelmap==i)))];
                    region_feas = [region_feas; tmp_fea];
                end
            end
            num_regions = cnt;
            % save results
            save(save_name, 'new_region_labelmap', 'region_feas', 'num_regions', 'new_region_size');
        end
        
        
        % use robust fit get flakes in the image, save each region, # of
        % flakes, contrast of each region
        % sum the residual of robustfit on multiple channels
        function process_one_img2(img_name, bk_img_name, bk_region_loc, save_name, fig_save_name, im_id)
            bk_im = imread(bk_img_name);
            bk_gray = rgb2gray(bk_im); bk_hsv = rgb2hsv(bk_im); bk_v = bk_hsv(:,:,3);
            im = imread(img_name);
            im_gray = rgb2gray(im); im_hsv = rgb2hsv(im); im_v = im_hsv(:,:,3);
%             imd = im - bk_im;
%             imd_gray = rgb2gray(imd); imd_hsv = rgb2hsv(imd);
%             [outlier_map_diff, region_labelmap, region_meanlocs, region_size, num_regions] = testdata_jan.perform_robustfit(double(im_gray(:,:,1)), im_gray);
            [outlier_map, residual_map, region_labelmap, region_meanlocs, region_size, num_regions] = testdata_jan.perform_robustfit_multichannel(im_hsv, im_gray, 10);
            % remove bk ground regions
            D = pdist2(region_meanlocs, bk_region_loc);
            D_min = min(D, [], 2);
            to_remove = find(D_min < 5);
            new_region_labelmap = zeros(size(region_labelmap));
            new_region_size = [];
            region_feas = [];
            cnt = 0;
            for i = 1: num_regions
                if ~ismember(i, to_remove)
                    cnt = cnt + 1;
                    new_region_labelmap(region_labelmap==i) = cnt;
                    new_region_size = [new_region_size; region_size(i)];
                    % get feature for the region
                    tmp_fea = [mean(double(im_gray(region_labelmap==i)) - double(bk_gray(region_labelmap==i))), ...
                        mean(double(im_v(region_labelmap==i)) - double(bk_v(region_labelmap==i)))];
                    region_feas = [region_feas; tmp_fea];
                end
            end
            num_regions = cnt;
            % save results
            save(save_name, 'residual_map', 'new_region_labelmap', 'region_feas', 'num_regions', 'new_region_size');
            figure(im_id);
            imshow(im);
            B = bwboundaries(new_region_labelmap>0, 'noholes');
            hold on;
            for k = 1:length(B)
               boundary = B{k};
               plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
            end
            print(im_id, '-dpng', [fig_save_name, '.png']);
        end
        
        function process_one_img2_getbw(img_name, bk_img_name, bk_region_loc, save_name, fig_save_name, im_id)
            load(save_name);
            bk_im = imread(bk_img_name);
            bk_gray = rgb2gray(bk_im); bk_hsv = rgb2hsv(bk_im); bk_v = bk_hsv(:,:,3);
            im = imread(img_name);
            im_gray = rgb2gray(im); im_hsv = rgb2hsv(im); im_v = im_hsv(:,:,3);
            
            shape_len_area_ratios = zeros(num_regions, 1);
            shape_contour_hists = zeros(num_regions, 15);
            shape_fracdims = zeros(num_regions, 1);
            
            stats_fea = zeros(num_regions, 16);
            stats_innerfea = zeros(num_regions, 16);
            stats_entropys = zeros(num_regions, 1);
            stats_innerentropys = zeros(num_regions, 1);
            [imH, imW, ~] = size(new_region_labelmap);
            se = strel('square', 5);
            for n = 1:num_regions
                bwmap = new_region_labelmap == n;
                contours = bwboundaries(bwmap, 'noholes');
                contours = contours{1};
                % compute shape information
                shape_len_area_ratios(n) = size(contours,1) / sum(bwmap(:));
                [fn,fr] = boxcount(bwmap,'slope');
                fdf = -diff(log(fn))./diff(log(fr));
                shape_fracdims(n) = mean(fdf(4:8));
                [r_r, r_c] = ind2sub([imH, imW], find(bwmap>0));
                bwcenter = [mean(r_r), mean(r_c)];
                % compute contour hists
                ct_dis = pdist2(bwcenter, contours);
                ct_hist = hist(ct_dis, 15);
                ct_hist = ct_hist / sum(ct_hist);
                shape_contour_hists(n,:) = ct_hist;
                
                % compute inner bwmap
                inner_bwmap = imerode(bwmap, se);
                
                % compute stats
                im_bw = double(im(bwmap));
                tmp_fea = [mean(double(im_gray(bwmap)) - double(bk_gray(bwmap))), ...
                        mean(double(im_v(bwmap)) - double(bk_v(bwmap))), ...
                        mean(double(im_gray(bwmap))), std(double(im_gray(bwmap)))];
                for c = 1: 3
                    im_c = double(im(:,:,c));
                    tmp_fea = [tmp_fea, mean(im_c(bwmap)), std(im_c(bwmap))];
                end
                for c = 1: 3
                    im_c = double(im_hsv(:,:,c));
                    tmp_fea = [tmp_fea, mean(im_c(bwmap)), std(im_c(bwmap))];
                end
                stats_fea(n, :) = tmp_fea;
                stats_entropys(n) = entropy(im_gray(bwmap));
                if sum(sum(inner_bwmap)) > 0
                    im_bw = double(im(inner_bwmap));
                    tmp_fea = [mean(double(im_gray(inner_bwmap)) - double(bk_gray(inner_bwmap))), ...
                            mean(double(im_v(inner_bwmap)) - double(bk_v(inner_bwmap))), ...
                            mean(double(im_gray(inner_bwmap))), std(double(im_gray(inner_bwmap)))];
                    for c = 1: 3
                        im_c = double(im(:,:,c));
                        tmp_fea = [tmp_fea, mean(im_c(inner_bwmap)), std(im_c(inner_bwmap))];
                    end
                    for c = 1: 3
                        im_c = double(im_hsv(:,:,c));
                        tmp_fea = [tmp_fea, mean(im_c(inner_bwmap)), std(im_c(inner_bwmap))];
                    end
                    stats_innerfea(n, :) = tmp_fea;
                    stats_innerentropys(n) = entropy(im_gray(inner_bwmap));
                end                    
            end
            % save results
            save(fig_save_name, 'shape_len_area_ratios', 'shape_contour_hists', 'shape_fracdims', 'stats_fea', 'stats_innerfea', 'stats_entropys', 'stats_innerentropys');
            
        end

        % region_meanlocs: [num_regions, 2]
        function [outlier_map, region_labelmap, region_meanlocs, region_size, num_regions] = perform_robustfit(im_c, im_gray, stats_fit)
%             mask = im_gray(:) > 20; % pixels which are not completely black. The mask is to avoid black border.
            mask = im_gray(:) > 0;
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
            outlier_map(mask) = abs(stats.resid) > stats_fit*stats.s;
            stats.s

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
            
            threshold = 0;%10*8;
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
%                     [r_r, r_c] = ind2sub([imH, imW], mean(CC.PixelIdxList{r_idx}));
                    [r_r, r_c] = ind2sub([imH, imW], CC.PixelIdxList{r_idx});
                    region_meanlocs = [region_meanlocs; mean(r_r), mean(r_c)];
                    region_size = [region_size; ccSzs(r_idx)];
                end
            end
            
            
        end
        
        % region_meanlocs: [num_regions, 2]
        function [outlier_map, res_map, region_labelmap, region_meanlocs, region_size, num_regions] = perform_robustfit_multichannel(im_hsv, im_gray, im_thre)
%             mask = im_gray(:) > 20; % pixels which are not completely black. The mask is to avoid black border.
            im_grayhs = cat(3, double(im_gray(:,:,1)), im_hsv(:,:,1:2));
%             im_grayhs = double(im_grayhs);
            mask = im_gray(:) > 0;
            [imH, imW, ~] = size(im_grayhs);
            [C, R] = meshgrid(0:(imW-1), 0:(imH-1));

            Y = R(mask)/(imH-1) - 0.5;
            X = C(mask)/(imW-1) - 0.5;

            A = [X.^2, Y.^2, X.*Y, X, Y];
            %             A = [(X.^2 + Y.^2)];
            res_map = zeros([imH, imW]);
            for c=1:3
                im_c = im_grayhs(:,:,c);
                [~,stats_c] = robustfit(A, im_c(mask));
                if c == 1
                    w = 1;
                    ref = stats_c.s;
                else
                    w = ref / stats_c.s;
                end
                res_map(mask) = res_map(mask) + w * abs(stats_c.resid);
            end
%             figure(3); imagesc(res_map)
            outlier_map = double(res_map>im_thre);
       
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
            
            threshold = 0;%10*8;
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
%                     [r_r, r_c] = ind2sub([imH, imW], mean(CC.PixelIdxList{r_idx}));
                    [r_r, r_c] = ind2sub([imH, imW], CC.PixelIdxList{r_idx});
                    region_meanlocs = [region_meanlocs; mean(r_r), mean(r_c)];
                    region_size = [region_size; ccSzs(r_idx)];
                end
            end
            
            
        end
        
        function L = perform_watershed(I)
%             figure(100);
            nr = 3;
            nc = 3;
            img_cnt = 1;
            gmag = imgradient(I);
%             subplot(nr,nc, img_cnt); imshow(I); title('Gray Image'); img_cnt = img_cnt + 1;
%             subplot(nr,nc, img_cnt); imshow(gmag,[]); title('Gradient Magnitude'); img_cnt = img_cnt + 1;
            L = watershed(gmag);
            Lrgb = label2rgb(L);
%             subplot(nr,nc, img_cnt); imshow(Lrgb); title('Watershed Transform of Gradient Magnitude'); img_cnt = img_cnt + 1;
            
            se = strel('disk',3);
%             Io = imopen(I,se);
%             subplot(nr,nc, 4); imshow(Io); title('Opening')
            
            Ie = imerode(I,se);
            Iobr = imreconstruct(Ie,I);
%             subplot(nr,nc, img_cnt); imshow(Iobr); title('Opening-by-Reconstruction'); img_cnt = img_cnt + 1;
            
%             Ioc = imclose(Io,se);
%             subplot(nr,nc, 6); imshow(Ioc); title('Opening-Closing')
            
            Iobrd = imdilate(Iobr,se);
            Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
            Iobrcbr = imcomplement(Iobrcbr);
%             subplot(nr,nc, img_cnt); imshow(Iobrcbr); title('Opening-Closing by Reconstruction'); img_cnt = img_cnt + 1;
            
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
%             subplot(nr,nc, img_cnt); imagesc(fgm4); title('Modified'); img_cnt = img_cnt + 1;
%             bw = imbinarize(Iobrcbr);
            bw = fgm;
%             level = graythresh(I);
%             bw = im2bw(I, level);
%             subplot(nr,nc, img_cnt); imshow(bw); title('Thresholded'); img_cnt = img_cnt + 1;
            
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
%             subplot(nr,nc, img_cnt); imagesc(labels); title('Markers & Boundaries'); img_cnt = img_cnt + 1;
%             subplot(nr,nc, img_cnt); imshow(Lrgb); title('Watershed Label Matrix'); img_cnt = img_cnt + 1;
        end
    end
end
