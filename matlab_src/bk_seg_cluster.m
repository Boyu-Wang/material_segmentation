imrgb = imread('01172018 G chip9.-000003.jpg');
imhsv = rgb2hsv(imrgb);
[imh, imw, ~] = size(imrgb);
%% use clustering to get background
[centers, assignments] = vl_kmeans(reshape(double(imrgb),[],3)', 10);
assignments = double(assignments);
num_occurs = hist(assignments, unique(assignments));
[~, background_idx] = max(num_occurs);
background_value = centers(:, background_idx);

discrete_im = reshape(assignments', imh, imw);
figure(1)
subplot(1,2,1);
imshow(imrgb);
subplot(1,2,2);
imagesc(discrete_im); axis image
print(1, '-dpdf', '../results/bk_seg_cluster/cluster_id.pdf')
%% use edge detection
edge_im = edge(rgb2gray(imrgb), 'canny', 10/255, 4);
figure(2);
subplot(1,2,1)
imshow(imrgb); axis image
subplot(1,2,2)
imshow(edge_im); axis image
print(2, '-dpdf', '../results/bk_seg_cluster/edge.pdf')