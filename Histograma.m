%HISTOGRAMAS (Valores absolutos)
% Configuración inicial
carpeta = ''; %direccion de la carpeta
extension = '*.png'; 

% Obtener lista de archivos
archivos = dir(fullfile(carpeta, extension));
numImagenes = length(archivos);

% Crear figura
figure('Name', 'Histogramas', 'NumberTitle', 'off', 'Position', [100, 100, 1400, 700]);
hold on;
grid on;

% Titulo y etiquetas
title('Histogramas (Valores Absolutos)');
xlabel('Nivel de intensidad (0-255)');
ylabel('Número de píxeles (Frecuencia Absoluta)');
xlim([0 255]);
set(gca, 'FontSize', 12);

% Colormap para diferenciar frames
colores = jet(numImagenes);

% Bucle p/c imagen
for i = 1:numImagenes
    imagen = imread(fullfile(carpeta, archivos(i).name));
    
    % Escala de grises
    %if size(imagen, 3) == 3
    %    imagen = rgb2gray(imagen);
    %end
    
    [counts, bins] = imhist(imagen);
    
    % Plot
    plot(bins, counts, 'Color', [colores(i,:), 0.3], 'LineWidth', 0.8);
end

% Colorbar
colorbar('Ticks', linspace(0, 1, 5), 'TickLabels', {'Frame inicial','','','', 'Frame final'});
colormap(jet);

hold off;

% figure('Name', 'Histograma Promedio', 'NumberTitle', 'off');
% histogramaPromedioAbs = zeros(256, 1);
% 
%  Obtener los bins de la primera imagen para usar como referencia
% imagenEjemplo = imread(fullfile(carpeta, archivos(1).name));
% if size(imagenEjemplo, 3) == 3
%     imagenEjemplo = rgb2gray(imagenEjemplo);
% end
% [~, bins] = imhist(imagenEjemplo);
% 
% for i = 1:numImagenes
%     imagen = imread(fullfile(carpeta, archivos(i).name));
%     if size(imagen, 3) == 3
%         imagen = rgb2gray(imagen);
%     end
%     counts = imhist(imagen);
%     histogramaPromedioAbs = histogramaPromedioAbs + counts / numImagenes;
% end
% 
% plot(bins, histogramaPromedioAbs, 'r', 'LineWidth', 2);
% grid on;
% title('Histograma promedio (Valores Absolutos)');
% xlabel('Nivel de intensidad (0-255)');
% ylabel('Número promedio de píxeles');
% xlim([0 255]);
