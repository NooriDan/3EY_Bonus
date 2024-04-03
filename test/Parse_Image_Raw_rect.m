clc
clear

img_raw = load("rect_WALLS.txt");

height = 480; width = 848;

img1 = zeros(height, width);

img2 = zeros(height, width);

% for i = 1:height
%     for j = 1:width
%         img1(i, j) = img_raw((i-1)*height + j);
%         img2(i, j) = img_raw(height*width + (i-1)*height + j);
%     end
% end

for j = 1:width
    for i = 1:height
        img1(i, j) = img_raw((i-1)*height + j);
        img2(i, j) = img_raw(height*width + (i-1)*height + j);
    end
end
