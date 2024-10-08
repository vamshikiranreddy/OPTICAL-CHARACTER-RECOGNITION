image = imread('your_image.png');
grayImage = rgb2gray(image);
adjustedImage = imadjust(grayImage);
bwImage = imbinarize(adjustedImage);
edges = edge(bwImage, 'Canny');
se = strel('disk', 2);
dilatedImage = imdilate(edges, se);
filledImage = imfill(dilatedImage, 'holes');
cleanedImage = bwareaopen(filledImage, 100);
[L, Ne] = bwlabel(cleanedImage);
propied = regionprops(L, 'BoundingBox', 'Area', 'Image', 'Centroid');
imshow(image);
hold on;
for n = 1:size(propied, 1)
    rectangle('Position', propied(n).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
    plot(propied(n).Centroid(1), propied(n).Centroid(2), 'bo');
end
numChars = Ne;
features = [];
labels = [];
for n = 1:Ne
    subImage = propied(n).Image;
    resizedImage = imresize(subImage, [42 24]);
    featureVector = extractHOGFeatures(resizedImage);
    features = [features; featureVector];
    labels = [labels; n];
end
trainedModel = fitcecoc(features, labels);
ocrResults = ocr(cleanedImage);
disp(ocrResults.Text);
gaussianFiltered = imgaussfilt(grayImage, 2);
medianFiltered = medfilt2(gaussianFiltered, [3 3]);
sharpenedImage = imsharpen(medianFiltered);
edgeDetected = edge(sharpenedImage, 'Sobel');
se2 = strel('square', 3);
dilatedEdgeImage = imdilate(edgeDetected, se2);
erodedImage = imerode(dilatedEdgeImage, se2);
finalBinaryImage = imbinarize(erodedImage);
filledBinaryImage = imfill(finalBinaryImage, 'holes');
connectedImage = bwareaopen(filledBinaryImage, 50);
[L2, Ne2] = bwlabel(connectedImage);
propied2 = regionprops(L2, 'BoundingBox', 'Area', 'Image', 'Centroid');
imshow(image);
hold on;
for n = 1:size(propied2, 1)
    rectangle('Position', propied2(n).BoundingBox, 'EdgeColor', 'g', 'LineWidth', 1);
    plot(propied2(n).Centroid(1), propied2(n).Centroid(2), 'ro');
end
features2 = [];
labels2 = [];
for n = 1:Ne2
    subImage2 = propied2(n).Image;
    resizedImage2 = imresize(subImage2, [42 24]);
    featureVector2 = extractHOGFeatures(resizedImage2);
    features2 = [features2; featureVector2];
    labels2 = [labels2; n];
end
trainedModel2 = fitcecoc(features2, labels2);
recognizedChars = [];
for n = 1:Ne2
    subImage2 = propied2(n).Image;
    resizedImage2 = imresize(subImage2, [42 24]);
    featureVector2 = extractHOGFeatures(resizedImage2);
    predictedLabel = predict(trainedModel2, featureVector2);
    recognizedChars = [recognizedChars, char(predictedLabel + 64)];
end
disp(['Recognized Characters: ', recognizedChars]);
croppedImage = imcrop(image, [50 50 300 300]);
rotatedImage = imrotate(croppedImage, 5);
resizedCroppedImage = imresize(rotatedImage, [512 512]);
alignedImage = imwarp(resizedCroppedImage, affine2d([1 0 0; 0 1 0; 50 50 1]));
finalAlignedImage = imresize(alignedImage, [1024 1024]);
imshow(finalAlignedImage);
ocrProcessedImage = ocr(finalAlignedImage);
recognizedTextFinal = ocrProcessedImage.Text;
disp(recognizedTextFinal);
glcm = graycomatrix(grayImage, 'Offset', [2 0; 0 2]);
stats = graycoprops(glcm, {'contrast', 'homogeneity'});
contrastValue = stats.Contrast;
homogeneityValue = stats.Homogeneity;
disp(['Contrast: ', num2str(contrastValue)]);
disp(['Homogeneity: ', num2str(homogeneityValue)]);
featuresFinal = [];
labelsFinal = [];
for n = 1:Ne2
    subImageFinal = propied2(n).Image;
    resizedImageFinal = imresize(subImageFinal, [42 24]);
    featureVectorFinal = extractHOGFeatures(resizedImageFinal);
    featuresFinal = [featuresFinal; featureVectorFinal];
    labelsFinal = [labelsFinal; n];
end
trainedModelFinal = fitcecoc(featuresFinal, labelsFinal);
recognizedCharsFinal = [];
for n = 1:Ne2
    subImageFinal = propied2(n).Image;
    resizedImageFinal = imresize(subImageFinal, [42 24]);
    featureVectorFinal = extractHOGFeatures(resizedImageFinal);
    predictedLabelFinal = predict(trainedModelFinal, featureVectorFinal);
    recognizedCharsFinal = [recognizedCharsFinal, char(predictedLabelFinal + 64)];
end
disp(['Final Recognized Characters: ', recognizedCharsFinal]);
thresholdedFinal = imbinarize(finalAlignedImage);
imshow(thresholdedFinal);
processedOCRResults = ocr(thresholdedFinal);
disp(processedOCRResults.Text);
noiseReducedImage = medfilt2(finalAlignedImage, [5 5]);
finalSharpenedImage = imsharpen(noiseReducedImage, 'Radius', 2, 'Amount', 1.5);
imshow(finalSharpenedImage);
finalEdges = edge(finalSharpenedImage, 'Canny');
finalDilatedEdges = imdilate(finalEdges, se);
finalFilledImage = imfill(finalDilatedEdges, 'holes');
finalCleanedImage = bwareaopen(finalFilledImage, 150);
[LFinal, NeFinal] = bwlabel(finalCleanedImage);
finalPropied = regionprops(LFinal, 'BoundingBox', 'Area', 'Image', 'Centroid');
imshow(image);
hold on;
for n = 1:size(finalPropied, 1)
    rectangle('Position', finalPropied(n).BoundingBox, 'EdgeColor', 'b', 'LineWidth', 1);
end
finalOcrResults = ocr(finalCleanedImage);
disp(finalOcrResults.Text);
