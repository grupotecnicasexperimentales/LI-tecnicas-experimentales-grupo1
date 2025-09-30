vid = VideoReader("Li Copernicus 45s.mp4");
frames = {};
frame_idx = 1;
peak_scores = [];  %  arreglo de peak
sharp_scores = [];  %   arreglo sharp
combined_scores = [];  %   arreglo combinado

while hasFrame(vid)
    frame = readFrame(vid);
    frames{frame_idx} = rgb2gray(frame);
    
    % Calcular la métrica de sharpness para el frame actual
    sharp_scores(frame_idx) = calculate_sharpness_metric(frames{frame_idx});
    
    % Definir cómo calcular la puntuación de "peak"
    peak_scores(frame_idx) = calculate_peak_metric(frames{frame_idx}); 
    
    % Métrica combinada (normalizada)
    peak_norm = normalize_score(peak_scores(frame_idx), peak_scores);
    sharp_norm = normalize_score(sharp_scores(frame_idx), sharp_scores);
    combined_scores(frame_idx) = 0.6 * peak_norm + 0.4 * sharp_norm;
    
    frame_idx = frame_idx + 1;
end

% --- Paso 2: Ranking de frames ---
[~, peak_rank] = sort(peak_scores, 'descend');
[~, sharp_rank] = sort(sharp_scores, 'descend');
[~, combined_rank] = sort(combined_scores, 'descend');

% --- Paso 3: Selección de mejores frames ---
N = length(frames);
top10 = round(0.10 * N);
top25 = round(0.25 * N);
top50 = round(0.50 * N);

best10_idx = combined_rank(1:top10);
best25_idx = combined_rank(1:top25);
best50_idx = combined_rank(1:top50);

% --- Paso 4: Apilado ---
stack10 = cat(3, frames{best10_idx});
result10 = mean(stack10, 3);

stack25 = cat(3, frames{best25_idx});
result25 = mean(stack25, 3);

stack50 = cat(3, frames{best50_idx});
result50 = mean(stack50, 3);


%% --- Visualización ---
figure; 
imshow(frames{combined_rank(1)}, []);  
title('mejor Frame Individual');
saveas(gcf, 'mejor_Frame_Individual.png');  % Guardar la imagen como PNG
close;

figure;  % Crear una nueva figura
imshow(result10, []);  % Mostrar el resultado del apilado Top 10%
title('Apilado Top 10%');
saveas(gcf, 'Apilado_Top_10.png');  % Guardar la imagen como PNG
close;

figure;  % Crear una nueva figura
imshow(result25, []);  % Mostrar el resultado del apilado Top 25%
title('Apilado Top 25%');
saveas(gcf, 'Apilado_Top_25.png');  % Guardar la imagen como PNG
close;

figure;  % Crear una nueva figura
imshow(result50, []);  % Mostrar el resultado del apilado Top 50%
title('Apilado Top 50%');
saveas(gcf, 'Apilado_Top_50.png');  % Guardar la imagen como PNG
close;

%% --- Funciones auxiliares ---
function sharpness = calculate_sharpness_metric(frame)
    h = fspecial('laplacian', 0.2);
    lap = imfilter(double(frame), h, 'replicate');
    sharpness = sum(abs(lap(:)));  % Suma de gradientes
end

function peak = calculate_peak_metric(frame)
    % Aquí deberías implementar cómo calcular la métrica de "pico"
    % Este es un ejemplo simple basado en la varianza de los píxeles:
    peak = var(double(frame(:)));
end

function val_norm = normalize_score(val, vec)
    % Normalización min-max
    val_norm = (val - min(vec)) / (max(vec) - min(vec));
end

