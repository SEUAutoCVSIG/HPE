%%
%   Created on Sept 23 10:09 2018
%
%   Author         : Shaoshu Yang
%   Email          : 13558615057@163.com
%   Last edit date : Sept 25 23:24 2018
%
%South East University Automation College, 211189 Nanjing China

% Get the length mpii dataset
len = size(RELEASE.annolist, 2);

file = fopen('train.txt', 'w');

% Traverse
for i = 1:len
    if RELEASE.img_train(i) == 1
        % Get number of people
        person_num = size(RELEASE.annolist(i).annorect, 2);
        
        for j = 1:person_num
            if  (~isfield(RELEASE.annolist(i).annorect(j), 'objpos')) || isempty(RELEASE.annolist(i).annorect(j).objpos)
                continue;
            end
            
            % File name
            filename = RELEASE.annolist(i).image.name;
            
            % Object position
            objcoord = zeros(1, 2);
            objcoord(1) = RELEASE.annolist(i).annorect(j).objpos.x;
            objcoord(2) = RELEASE.annolist(i).annorect(j).objpos.y;
            
            % Object scale
            scale = RELEASE.annolist(i).annorect(j).scale;
            
            % Key points
            key_points = zeros(1, 32);
            for k = 0:15
                pos = find([RELEASE.annolist(i).annorect(j).annopoints.point.id] == k);
                
                if isempty(pos)
                    key_points(2*k + 1) = 0;
                    key_points(2*k + 2) = 0;
                else
                    key_points(2*k + 1) = RELEASE.annolist(i).annorect(j).annopoints.point(pos).x;
                    key_points(2*k + 2) = RELEASE.annolist(i).annorect(j).annopoints.point(pos).y;
                end
            end
            
            fprintf(file, "%s ", filename);
            fprintf(file, "%d ", objcoord);
            fprintf(file, "%f ", scale);
            fprintf(file, "%d ", int32(key_points));
            fprintf(file, "\r\n");
        end
    end
end

fclose(file);
