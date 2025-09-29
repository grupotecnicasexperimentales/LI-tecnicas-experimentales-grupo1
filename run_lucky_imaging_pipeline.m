% run_lucky_imaging_pipeline.m
% Script para ejecutar lucky_imaging_pipeline con opciones y diagnósticos.
clear; clc; close all;

%% === CONFIGURACIÓN ===
video_file = 'jupiterLi.mp4';   % nombre del archivo de video 
use_video_file = true;                  % true: pasa el filename; false: carga frames manualmente y pasa array
process_subset = false;                 % true: cargar solo un subconjunto de frames (rápido para pruebas)
subset_range = 1:200;                   % si process_subset = true

% Parámetros del pipeline
params.selection_percentage = 10;       % % de frames a seleccionar 
params.selection_method = 'sharpness';   % 'peak'|'sharpness'|'combined'
params.stacking_method = 'mean';    % 'mean'|'median'|'weighted'
params.enhancement_method = 'unsharp';  % 'unsharp'|'none'
params.alignment = true;

% Opciones de rendimiento/diagnóstico
show_diagnostics = true;                % muestra figuras y gráficos
save_results = true;                    % guarda imágenes resultado
output_prefix = 'lucky_run_01';


%% === CARGAR / PREPARAR INPUT ===
if use_video_file
    fprintf('Usando archivo de video: %s\n', video_file);
    if process_subset
        % Cargar sólo subset_range frames usando VideoReader
        vr = VideoReader(video_file);
        totalFrames = floor(vr.Duration * vr.FrameRate);
        subset_range = subset_range(subset_range <= totalFrames);
        fprintf('Cargando %d frames (subset)...\n', numel(subset_range));
        frames_cell = cell(1, numel(subset_range));
        idx = 1;
        for k = subset_range
            vr.CurrentTime = (k-1) / vr.FrameRate;
            f = readFrame(vr);
            if size(f,3) == 3
                try
                    f = rgb2gray(f);
                catch
                    f = uint8(0.2989*double(f(:,:,1)) + 0.5870*double(f(:,:,2)) + 0.1140*double(f(:,:,3)));
                end
            end            
            frames_cell{idx} = f;
            idx = idx + 1;
        end
        frames = cat(3, frames_cell{:});
        input_for_pipeline = frames;
        use_video_file = false; 
    else
        % pasar directamente el filename al pipeline (dejar que la función lo lea)
        input_for_pipeline = video_file;
    end
else
    % Si ya tienes un arreglo frames en workspace, asignarlo aquí:
    % frames = yourLoadedFrames;  % rows x cols x N
    % Si quieres cargar desde file, puedes usar VideoReader como arriba.
    error('set use_video_file = true o carga frames manualmente en este script.');
end

%% === EJECUTAR PIPELINE Y MEDIR TIEMPO ===
fprintf('Iniciando pipeline...\n');
tstart = tic;
result = lucky_imaging_pipeline(input_for_pipeline, params); %ejecuta el archivo lucky_imaging_pipeline
ttotal = toc(tstart);
fprintf('Pipeline finalizado en %.2f s.\n', ttotal);

%% === DIAGNÓSTICOS Y VISUALIZACIONES ===
if show_diagnostics
    figure('Name','Resultado Lucky','NumberTitle','off');
    subplot(1,2,1);
    imshow(result.baseline, []); title('Baseline (mean)');
    subplot(1,2,2);
    imshow(result.image, []); title(sprintf('Lucky (%.0f%% frames)', params.selection_percentage));

    % Mostrar algunos de los frames seleccionados (si tenemos acceso a frames)
    try
        if ~ischar(input_for_pipeline) && ~isstring(input_for_pipeline)
            % tenemos el arreglo frames en memoria
            allFrames = input_for_pipeline;
            idx_sel = result.selected_frames;
            nshow = min(8, numel(idx_sel));
            figure('Name','Mejores frames (montage)','NumberTitle','off');
            montage(mat2gray(allFrames(:,:,idx_sel(1:nshow))));
            title('Mejores frames seleccionados');
        else
            % si pasamos filename, no tenemos frames cargadas; mostrar aviso
            fprintf('No hay arreglo de frames en memoria para mostrar montaje (se pasó filename).\n');
        end
    catch ME
        warning('No se generó montaje de frames: %s', ME.message);
    end

    % Si result.performance existe, mostrar métricas
    if isfield(result, 'performance')
        fprintf('Performance: resolution_improvement=%.3f, snr_ratio=%.3f, efficiency=%.3f\n', ...
            result.performance.resolution_improvement, ...
            result.performance.snr_ratio, ...
            result.performance.efficiency);
    end

end

%% === GUARDAR RESULTADOS ===
if save_results
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    out_base = sprintf('%s_%s', output_prefix, timestamp);

    % Guardar imagen apilada (convertir a uint16/uint8 según rango)
    out_img = result.image;
    % normalizar al rango 0-1 y guardar como PNG 16-bit para calidad
    out_norm = out_img - min(out_img(:));
    out_norm = out_norm / max(out_norm(:));
    imwrite(im2uint16(out_norm), [out_base '_lucky.tif']);  % TIFF 16-bit
    imwrite(im2uint8(out_norm),  [out_base '_lucky_u8.png']);
    % Guardar baseline
    baseline = result.baseline;
    bnorm = baseline - min(baseline(:)); bnorm = bnorm / max(bnorm(:));
    imwrite(im2uint16(bnorm), [out_base '_baseline.tif']);

    % Guardar resultados de estructura
    save([out_base '_result.mat'], 'result');

    fprintf('Resultados guardados con prefijo: %s\n', out_base);
end


