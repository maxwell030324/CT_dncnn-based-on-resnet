vld_tgt_location = './vld_tgt';
vld_raw_location = './vld_raw';
train_tgt_location = './train_tgt';
train_raw_location = './train_raw';
test_tgt_location = './test_tgt';
test_raw_location = './test_raw';

subdir = dir(train_tgt_location);

for i = 1:length(subdir)
    if( isequal( subdir( i ).name, '.' ) ||  isequal( subdir( i ).name, '..' ))   % 如果不是目录 就跳过
        continue;
    end
%     if i~=3
%         continue;
%     end
    picpath = fullfile(train_tgt_location, subdir(i).name);
    savepath = fullfile(train_raw_location, subdir(i).name);
    pic_mtx = imread(picpath);

    %disp(pic_mtx)
    theta3 = 0:1:179.9;

    [R3,~] = radon(pic_mtx, theta3);
    J3 = iradon(R3,theta3);
    pic_raw = imnoise(uint8(J3),"poisson");
    imwrite(pic_raw,savepath)
%     figure,
%     subplot(121), imshow(uint8(pic_mtx)), title('origion');
%     subplot(122), imshow(uint8(J3)), title('N=900');
end

