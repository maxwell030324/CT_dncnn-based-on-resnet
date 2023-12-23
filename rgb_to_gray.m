exp_tgt_location = 'E:\\核医学物理原理\\课程大作业\\exp_tgt';
exp_raw_location = 'E:\\核医学物理原理\课程大作业\\exp_raw';
train_tgt_location = 'E:\\核医学物理原理\课程大作业\\train_tgt';
train_raw_location = 'E:\\核医学物理原理\课程大作业\\train_raw';
test_tgt_location = 'E:\\核医学物理原理\课程大作业\\test_tgt';
test_raw_location = 'E:\\核医学物理原理\课程大作业\\test_raw';

subdir = dir(test_tgt_location);

for i = 1:length(subdir)
    if( isequal( subdir( i ).name, '.' ) ||  isequal( subdir( i ).name, '..' ))   % 如果不是目录 就跳过
        continue;
    end
%     if i~=3
%         continue;
%     end
    picpath = fullfile(test_tgt_location, subdir(i).name);
    savepath = fullfile(test_raw_location, subdir(i).name);
    pic_mtx = imread(picpath);
    if ndims(pic_mtx)==3
        pic_mtx = rgb2gray(pic_mtx); % 将矩阵转为二维
    end
    
    imwrite(pic_mtx,picpath)
%     figure,
%     subplot(121), imshow(uint8(pic_mtx)), title('origion');
%     subplot(122), imshow(uint8(J3)), title('N=900');
end