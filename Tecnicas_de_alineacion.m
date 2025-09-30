%% Comparación de técnicas de alineación
vid = VideoReader("Li Copernicus 45s.mp4");

frames = {};
frame_idx = 1;
while hasFrame(vid)
    frame = readFrame(vid);
    frames{frame_idx} = rgb2gray(frame);
    frame_idx = frame_idx + 1;
end

% Convertir a arreglo 3D
video_frames = cat(3, frames{:});

% Ranking y selección (ejemplo: top 10%)
[ranked_frames, quality_scores] = rank_frames(video_frames, 'combined');
N = round(0.1 * size(video_frames,3));
best_idx = ranked_frames(1:N);

% Frame de referencia
ref = double(video_frames(:,:,best_idx(1)));

%% --- 1. Sin alineación ---
stack_none = cat(3, video_frames(:,:,best_idx));
result_none = mean(stack_none, 3); 

%% --- 2. Alineación por correlación cruzada ---
aligned_corr = cell(1,N);
for i = 1:N
    moving = double(video_frames(:,:,best_idx(i)));
    c = normxcorr2(moving, ref);
    [~, maxIdx] = max(c(:));
    [ypeak, xpeak] = ind2sub(size(c), maxIdx);
    offset = [xpeak - size(moving,2), ypeak - size(moving,1)];
    tform = affine2d([1 0 0; 0 1 0; offset(1) offset(2) 1]);
    aligned_corr{i} = imwarp(moving, tform, 'OutputView', imref2d(size(ref)));
end
stack_corr = cat(3, aligned_corr{:});
result_corr = mean(stack_corr, 3);

%% --- 3. Alineación por centroide (pico de intensidad) ---
aligned_cent = cell(1,N);
stats_ref = regionprops(ref > 0.5*max(ref(:)), ref, 'WeightedCentroid');
cent_ref = stats_ref(1).WeightedCentroid;
for i = 1:N
    moving = double(video_frames(:,:,best_idx(i)));
    stats = regionprops(moving > 0.5*max(moving(:)), moving, 'WeightedCentroid');
    if isempty(stats), aligned_cent{i} = moving; continue; end
    cent = stats(1).WeightedCentroid;
    shift = cent_ref - cent;
    tform = affine2d([1 0 0; 0 1 0; shift(1) shift(2) 1]);
    aligned_cent{i} = imwarp(moving, tform, 'OutputView', imref2d(size(ref)));
end
stack_cent = cat(3, aligned_cent{:});
result_cent = mean(stack_cent, 3);

%% --- 4. Interpolación bicúbica (aplicada a correlación cruzada) ---
aligned_bic = cell(1,N);
for i = 1:N
    moving = double(video_frames(:,:,best_idx(i)));
    c = normxcorr2(moving, ref);
    [~, maxIdx] = max(c(:));
    [ypeak, xpeak] = ind2sub(size(c), maxIdx);
    offset = [xpeak - size(moving,2), ypeak - size(moving,1)];
    tform = affine2d([1 0 0; 0 1 0; offset(1) offset(2) 1]);
    aligned_bic{i} = imwarp(moving, tform, ...
        'OutputView', imref2d(size(ref)), ...
        'InterpolationMethod', 'bicubic');
end
stack_bic = cat(3, aligned_bic{:});
result_bic = mean(stack_bic, 3);

%% --- Mostrar resultados ---
figure;
subplot(2,2,1);
imshow(uint8(result_none), []);
title('Sin alineación');

subplot(2,2,2);
imshow(uint8(result_corr), []);
title('Correlación cruzada');

subplot(2,2,3);
imshow(uint8(result_cent), []);
title('Centroide / pico de intensidad');

subplot(2,2,4);
imshow(uint8(result_bic), []);
title('Interpolación bicúbica sub-píxel');

%% --- Función de ranking de frames ---
function [ranked_frames, quality_scores] = rank_frames(video_frames, criterion)
    num_frames = size(video_frames, 3);
    quality_scores = zeros(num_frames, 1);

    for i = 1:num_frames
        frame = video_frames(:,:,i);

        switch criterion
            case 'peak'
                quality_scores(i) = max(frame(:));

            case 'sharpness'
                quality_scores(i) = calculate_sharpness_metric(frame);

            case 'combined'
                peak_score  = max(frame(:));
                sharp_score = calculate_sharpness_metric(frame);

                % Normalización (para evitar escalas distintas)
                peak_norm  = peak_score / (max(frame(:)) + eps);
                sharp_norm = sharp_score / (max(sharp_score,eps));

                quality_scores(i) = 0.6*peak_norm + 0.4*sharp_norm;

            otherwise
                error('Criterio no válido. Use "peak", "sharpness" o "combined".');
        end
    end

    % Orden descendente
    [~, sorted_indices] = sort(quality_scores, 'descend');
    ranked_frames = sorted_indices;
end

%% --- Función auxiliar: métrica de nitidez (Laplaciano) ---
function sharpness = calculate_sharpness_metric(frame)
    h = fspecial('laplacian', 0.2);
    lap = imfilter(double(frame), h, 'replicate');
    sharpness = sum(abs(lap(:)));
end
