clear
clc


%image_rect_raw = load("rect_WALLS.txt");
%image_rect_raw = load("image_rect_raw.txt");
image_rect_raw = load("Ready_Data/April2_box60cm.txt");


height = 480;
width = 848;

max_distance = 2000;            % in mm
min_noneZero_distance = 100;    % in mm

roi_x_lower = 100;
roi_x_upper = 600;
roi_y_lower = 150;
roi_y_upper = 400;

markerSize = 1;

depth_image = zeros(height, width);
depth_image_ROI = zeros(roi_y_upper - roi_y_lower +1,roi_x_upper - roi_x_lower +1);

%data is stored in big-endian
for i = 1:height
    for j = 1:width
        lowByte = image_rect_raw((i-1)*2*width + 2*(j-1)+1);
        highByte = image_rect_raw((i-1)*2*width + 2*j);
        depth_image(i,j) = highByte*2^8 + lowByte;
        
        % optional: clip max distance
        if(depth_image(i,j) ~= 0) % given a valid depth pixel
            if(depth_image(i,j) > max_distance)
                %depth_image(i,j) = max_distance; %clip
                depth_image(i,j) = 0; %remove
                
            elseif (depth_image(i,j) < min_noneZero_distance)
                depth_image(i,j) = 0;

            end
        end

        if(i<=roi_y_upper && i>=roi_y_lower && j>=roi_x_lower && j<=roi_x_upper)
            depth_image_ROI(i-roi_y_lower+1,j-roi_x_lower+1) = depth_image(i,j);
        end
    end
end

% Optional - Image processing

%depth_image = imfill(depth_image,"holes");
%depth_image_ROI = imfill(depth_image_ROI,"holes");

% Apply Gaussian smoothing
% sigma = 10; % Standard deviation of the Gaussian filter
% depth_image = imgaussfilt(im2double(depth_image), sigma);

subplot(2,2,1);
imshow(depth_image);

% Draw the rectangle
rectangle('Position', [roi_x_lower, roi_y_lower, roi_x_upper-roi_x_lower, roi_y_upper-roi_y_lower], 'EdgeColor', 'r', 'LineWidth', 2);

subplot(2,2,3);
imshow(depth_image_ROI);


% Camera Info
distortion_model = "plumb_bob";
D = [0 0 0 0 0]; % Distortion coefficient
K = [421.7 0 425.8759765625; 0 421.7 239.40527; 0 0 1]; %Intrinsic camera matrix - containing focal lengths and principal point
R = [1 0 0; 0 1 0; 0 0 1 ]; % rectification matrix
P = [421.7 0 425.8759765625; 0 0 421.7006; 239.40527 0 0; 0 1 0];
K_inv = inv(K);

fx = K(1);
fy = K(5);
cx = K(3);
cy = K(6);

x_cam = zeros(1, height*width);
y_cam = zeros(1, height*width);
z_cam = zeros(1, height*width);
index = 1;

% Project the depth and pixels to xyz coordinates (inefficient)
% for v = 1:height
%     for u = 1:width
%         P_frame = [u v 1];
%         z = depth_image(v, u);
%         P_cam = K_inv*(z*P_frame');
%         x_cam(index) = P_cam(1);
%         y_cam(index) = P_cam(2);
%         z_cam(index) = P_cam(3);
%         index = index +1;
%     end
% end

% % Project the depth and pixels to xyz coordinates (efficient)
for v = 1:height
    for u = 1:width
        z = depth_image(v, u);
        x_cam(index) = (z/fx)*(u-cx);
        y_cam(index) = (z/fy)*(v-cy);
        z_cam(index) = z;
        index = index +1;
    end
end

subplot(2,2,2)
scatter3(x_cam, y_cam, z_cam, markerSize)
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D point cloud - Global');

x_cam_ROI = zeros(1, (roi_y_lower-roi_y_upper+1)*(roi_x_lower-roi_x_upper+1));
y_cam_ROI = zeros(1, (roi_y_lower-roi_y_upper+1)*(roi_x_lower-roi_x_upper+1));
z_cam_ROI = zeros(1, (roi_y_lower-roi_y_upper+1)*(roi_x_lower-roi_x_upper+1));
% Project the depth and pixels to xyz coordinates (inefficient)
for v = roi_y_lower:roi_y_upper
    for u = roi_x_lower:roi_x_upper
        P_frame = [u v 1];
        z = depth_image(v, u);
        P_cam = K_inv*(z*P_frame');
        x_cam_ROI(index) = P_cam(1);
        y_cam_ROI(index) = P_cam(2);
        z_cam_ROI(index) = P_cam(3);
        index = index +1;
    end
end

% % Project the depth and pixels to xyz coordinates (efficient)
% for v = roi_y_lower:roi_y_upper
%     for u = roi_x_lower:roi_x_upper
%         z = depth_image(v, u);
%         x_cam_ROI(index) = (z/fx)*(u-cx);
%         y_cam_ROI(index) = (z/fy)*(v-cy);
%         z_cam_ROI(index) = z;
%         index = index +1;
%     end
% end

subplot(2,2,4)
scatter3(x_cam_ROI, y_cam_ROI, z_cam_ROI, markerSize)
%surf(x_cam_ROI, y_cam_ROI, z_cam_ROI, markerSize)

xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D point cloud - ROI');

showaxes('on')