
clear
clc
cla

image_rect_raw = load("image_rect_raw.txt");


height = 480;
width = 848;

max_distance = 2000;            % in mm
min_noneZero_distance = 300;    % in mm

depth_image = zeros(height, width);

%data is stored in big-endian

for i = 1:height
    for j = 1:width
        lowByte = image_rect_raw((i-1)*2*width + 2*(j-1)+1);
        highByte = image_rect_raw((i-1)*2*width + 2*j);
        depth_image(i,j) = highByte*2^8 + lowByte;
        
        % optional: clip max distance
        if(depth_image(i,j) ~= 0) % given a valid depth pixel
            if(depth_image(i,j) > max_distance)
                depth_image(i,j) = max_distance;

            elseif (depth_image(i,j) < min_noneZero_distance)
                depth_image(i,j) = 0;
            end
        end

    end
end

depth_image = imfill(depth_image,"holes");

% % show the pre-processed depth image
 imshow(depth_image)

% find the closest points
closestInCol = zeros(1, width);

% Create a mask for non-zero elements
nonzero_mask = (depth_image ~= 0);

% Apply the mask to depth_image
masked_depth_image = depth_image(nonzero_mask);

meanNoneZero = mean(masked_depth_image);
countNoneZero = max(size(masked_depth_image));


% convert the depth to point cloud
x = zeros(1, countNoneZero);
y = zeros(1, countNoneZero);
z = zeros(1, countNoneZero);

% Definig the coordinate frame
center_height = height/2;
center_width = width/2;

FOV_horizontal  = 40; % in deg
FOV_vertical    = 30; % in deg

% FOV_horizontal_step = FOV_horizontal/width;
% FOV_vertical_step = FOV_vertical/height;

FOV_horizontal_vector = deg2rad(linspace(-FOV_horizontal/2, FOV_horizontal/2, width));
FOV_vertical_vector   = deg2rad(linspace(-FOV_vertical/2, FOV_vertical/2, height));


for j = 1: width
    lastMin = inf;

    for i = 1:height

        if((depth_image(i,j) ~= 0) && (depth_image(i,j) < lastMin))
            lastMin = depth_image(i,j);
        end
        
        if(depth_image(i,j) ~= 0)
           
            theta   = FOV_horizontal_vector(j);
            phi     = FOV_vertical_vector(i);
            r       = depth_image(i,j);

            % Convert spherical coordinates to Cartesian coordinates
            x(i+j-1) = r * sin(theta) * cos(phi);
            y(i+j-1) = r * sin(theta) * sin(phi);
            z(i+j-1) = r * cos(theta); 
        end

    end
    closestInCol(j) = lastMin;
end


% Image processing
% imshow(depth_image)

% filled_img = imfill(depth_image,"holes");
% imshow(filled_img)


% % Create a scatter plot in 3D
% scatter3(x, y, z, 'filled');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% title('3D point cloud');

% % Draw the square on the image
% hold on;
% rectangle('Position', [x1, y1, width, height], 'EdgeColor', 'r', 'LineWidth', 2);
% hold off;


