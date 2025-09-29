function lucky_result = lucky_imaging_pipeline(vid, params) %funcion principal
% LUCKY_IMAGING_PIPELINE  Pipeline completo de Lucky Imaging. 
% utilizar run_lucky_imaging_pipeline.m para ejecutar el codigo
%   lucky_result = lucky_imaging_pipeline(vid, params)
%   vid: nombre de archivo de video (string) o arreglo 3D (rows x cols x N)
%   params: estructura con campos opcionales:
%       - selection_percentage (default 25)
%       - selection_method ('peak'|'sharpness'|'combined') (default 'combined')
%       - stacking_method ('weighted'|'mean'|'median') (default 'weighted')
%       - enhancement_method ('unsharp'|'none') (default 'unsharp')
%       - alignment (true|false) (default true)
 

    % --- Parámetros por defecto 
    %para cambiar los parametros utilizar run_lucky_imaging_pipeline.m
    if nargin < 2 || isempty(params)
        params = struct();
    end
    if ~isfield(params, 'selection_percentage'), params.selection_percentage = 25; end
    if ~isfield(params, 'selection_method'),    params.selection_method = 'combined'; end
    if ~isfield(params, 'stacking_method'),     params.stacking_method = 'weighted'; end
    if ~isfield(params, 'enhancement_method'),  params.enhancement_method = 'unsharp'; end
    if ~isfield(params, 'alignment'),           params.alignment = true; end

    % --- Cargar datos
    if ischar(vid) || isstring(vid)
        video_file = char(vid);
        if ~exist(video_file, 'file')
            error('Archivo de video no encontrado: %s', video_file);
        end
        fprintf('Cargando video: %s\n', video_file);
        frames = load_video_frames(video_file);
    else
        frames = vid;
    end

    % Validar frames
    if isempty(frames) || ndims(frames) ~= 3
        error('frames debe ser un arreglo 3D (rows x cols x N) no vacío.');
    end

    num_total = size(frames, 3);
    fprintf('Procesando %d frames...\n', num_total);

    % --- Paso 1: Ranking de frames
    fprintf('Ranking frames por calidad (%s)...\n', params.selection_method);
    [ranked_indices, quality_scores] = rank_frames(frames, params.selection_method);

    % --- Paso 2: Selección de mejores frames
    fprintf('Seleccionando mejor %d%% de frames...\n', params.selection_percentage);
    selected_indices = select_best_frames(ranked_indices, params.selection_percentage);
    selected_frames = frames(:,:,selected_indices);

    % --- Paso 2.5: Alineamiento  de frames seleccionados
    if params.alignment
        fprintf('Alineando %d frames seleccionados...\n', size(selected_frames,3));
        aligned_frames = align_frames(selected_frames);
    else
        aligned_frames = selected_frames;
    end

    % --- Paso 3: Apilamiento
    % obtener pesos desde quality_scores (correspondientes a índices originales)
    weights = quality_scores(selected_indices);
    if strcmpi(params.stacking_method, 'weighted')
        if all(weights==0)
            warning('Todos los pesos son cero — usando pesos uniformes.');
            weights = ones(size(weights));
        end
    end
    stacked_image = stack_images(aligned_frames, params.stacking_method, weights);

    % --- Paso 4: Imagen baseline (promedio de todos los frames)
    baseline_image = mean(double(frames), 3);

    % --- Paso 5: 
    switch lower(params.enhancement_method)
        case 'unsharp'
            % imsharpen  Image Processing Toolbox
            try
                enhanced_image = imsharpen(stacked_image);
            catch
                warning('imsharpen no disponible o fallo: devolviendo stacked_image sin realzar.');
                enhanced_image = stacked_image;
            end
        otherwise
            enhanced_image = stacked_image;
    end

    % --- Paso 6: Analizar rendimiento
    performance = analyze_performance(baseline_image, enhanced_image);

    % --- Guardar resultados
    lucky_result.image = enhanced_image;
    lucky_result.baseline = baseline_image;
    lucky_result.performance = performance;
    lucky_result.params = params;
    lucky_result.selected_frames = selected_indices;
    lucky_result.num_processed = length(selected_indices);

    fprintf('Procesamiento completado.\n');
    if ~isnan(performance.resolution_improvement)
        fprintf('Mejora en resolución: %.2fx\n', performance.resolution_improvement);
    else
        fprintf('Mejora en resolución: indefinida (NaN)\n');
    end
end

%% -----------------------
% Funciones auxiliares que se utilizan en la principal
%% -----------------------

function frames = load_video_frames(video_file)
    % Cargar frames desde archivo de video
    vidObj = VideoReader(video_file);
    frames_cell = {};
    idx = 1;
    while hasFrame(vidObj)
        f = readFrame(vidObj);
        if size(f,3) == 3
            % rgb2gray requiere Image Processing Toolbox; 
            try
                f = rgb2gray(f);
            catch
                f = uint8(0.2989*double(f(:,:,1)) + 0.5870*double(f(:,:,2)) + 0.1140*double(f(:,:,3)));
            end
        end
        frames_cell{idx} = f;
        idx = idx + 1;
    end
    if isempty(frames_cell)
        frames = [];
    else
        frames = cat(3, frames_cell{:});
    end
end

function selected = select_best_frames(ranked_indices, percentage)
    N = length(ranked_indices);
    num_select = max(1, round(N * percentage / 100));
    selected = ranked_indices(1:num_select);
end

function [ranked_frames, quality_scores] = rank_frames(video_frames, criterion)
    num_frames = size(video_frames, 3);
    quality_scores = zeros(num_frames, 1);
    for i = 1:num_frames
        frame = double(video_frames(:,:,i));
        switch lower(criterion)
            case 'peak'
                quality_scores(i) = max(frame(:));
            case 'sharpness'
                quality_scores(i) = calculate_sharpness_metric(frame);
            case 'combined'
                peak_score = max(frame(:));
                sharp_score = calculate_sharpness_metric(frame);
                quality_scores(i) = 0.6 * peak_score + 0.4 * sharp_score;
            otherwise
                error('Criterio de selección desconocido: %s', criterion);
        end
    end
    % Normalizar 
    if max(quality_scores) > 0
        quality_scores = quality_scores / max(quality_scores);
    end
    [~, sorted_indices] = sort(quality_scores, 'descend');
    ranked_frames = sorted_indices;
end

function sharp = calculate_sharpness_metric(frame)
    % Varianza del Laplaciano 
    try
        h = fspecial('laplacian', 0.2);
        lap = imfilter(double(frame), h, 'replicate');
        sharp = var(lap(:));
    catch
        % fallback: variación del gradiente
        [gx, gy] = gradient(double(frame));
        sharp = var(gx(:) + gy(:));
    end
end

function stacked_image = stack_images(aligned_frames, method, weights)
    switch lower(method)
        case 'mean'
            stacked_image = mean(double(aligned_frames), 3);
        case 'weighted'
            if nargin < 3 || isempty(weights)
                error('Pesos requeridos para apilamiento ponderado');
            end
            weights = double(weights(:));
            weights = weights / sum(weights);
            stacked_image = zeros(size(aligned_frames(:,:,1)));
            for i = 1:size(aligned_frames, 3)
                stacked_image = stacked_image + double(aligned_frames(:,:,i)) * weights(i);
            end
        case 'median'
            stacked_image = median(double(aligned_frames), 3);
        otherwise
            error('Método de apilamiento no reconocido: %s', method);
    end
end

function aligned = align_frames(frames3D)
    % Intentar alinear usando imregtform . Si falla, fallback a
    % normxcorr2. Si todo falla, devuelve los frames sin alinear.
    ref = double(frames3D(:,:,1));
    [h, w, N] = size(frames3D);
    aligned = zeros(h, w, N);
    aligned(:,:,1) = ref;
    % preparar opciones para imregtform si disponible
    use_imreg = exist('imregtform', 'file') == 2;
    if use_imreg
        try
            [optimizer, metric] = imregconfig('monomodal');
            for i = 2:N
                moving = double(frames3D(:,:,i));
                try
                    tform = imregtform(moving, ref, 'translation', optimizer, metric);
                    Rout = imref2d(size(ref));
                    registered = imwarp(moving, tform, 'OutputView', Rout);
                    aligned(:,:,i) = registered;
                catch
                    aligned(:,:,i) = moving; % fallback individual
                end
            end
            return;
        catch
            % si algo falla con imregtform, seguimos a fallback
        end
    end

    % Fallback: usar normxcorr2 (requiere Image Processing Toolbox)
    use_normxcorr = exist('normxcorr2', 'file') == 2;
    if use_normxcorr
        for i = 2:N
            moving = double(frames3D(:,:,i));
            try
                c = normxcorr2(ref, moving);
                [ypeak, xpeak] = find(c == max(c(:)), 1);
                yoff = ypeak - size(ref,1);
                xoff = xpeak - size(ref,2);
                tform = affine2d([1 0 0; 0 1 0; xoff yoff 1]);
                registered = imwarp(moving, tform, 'OutputView', imref2d(size(ref)));
                aligned(:,:,i) = registered;
            catch
                aligned(:,:,i) = moving;
            end
        end
        return;
    end

    % Si no hay herramientas disponibles, devolver tal cual y avisar
    warning('No se pudo alinear: imregtform y normxcorr2 no disponibles. Devolviendo frames sin alinear.');
    aligned = frames3D;
end

function performance = analyze_performance(original_image, lucky_image)
    fwhm_original = measure_image_fwhm(original_image);
    fwhm_lucky = measure_image_fwhm(lucky_image);
    if isnan(fwhm_original) || isnan(fwhm_lucky) || fwhm_lucky == 0
        performance.resolution_improvement = NaN;
    else
        performance.resolution_improvement = fwhm_original / fwhm_lucky;
    end

    snr_original = calculate_snr(original_image);
    snr_lucky = calculate_snr(lucky_image);
    if snr_original == 0
        performance.snr_ratio = NaN;
    else
        performance.snr_ratio = snr_lucky / snr_original;
    end

    if isnan(performance.resolution_improvement) || isnan(performance.snr_ratio)
        performance.efficiency = NaN;
    else
        performance.efficiency = performance.resolution_improvement * performance.snr_ratio;
    end
end

function fwhm_val = measure_image_fwhm(img)
    img = double(img);
    % usar FFT para estimar ancho efectivo; si falla, devolver NaN
    try
        psf = abs(fftshift(fft2(img)));
        psf = psf / max(psf(:));
        line = psf(round(end/2), :);
        idx = find(line <= 0.5, 1);
        if isempty(idx)
            fwhm_val = NaN;
        else
            fwhm_val = 2 * idx;
        end
    catch
        fwhm_val = NaN;
    end
end

function snr_val = calculate_snr(img)
    imgv = double(img(:));
    signal = mean(imgv);
    noise = std(imgv);
    snr_val = signal / (noise + eps);
end
