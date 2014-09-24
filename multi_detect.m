function demo()

name = 'G:/EDU/_SOURCE_CODE/Belgium/01/image.004639.jp2';
im=imread(name);
box1 = test(name,'sign');
box2 = test(name,'blueSign');
box1 = [box1;box2];
showboxes(im, box1);

function bbox = test(name, cls)
% load and display image
im=imread(name);
clf;
image(im);
axis equal; 
axis on;
disp('input image');
% disp('press any key to continue'); pause;

% load and display model
load(['VOC2007/' cls '_final']);
% load([cls '_final']);
% visualizemodel(model);
disp([cls ' model']);
% disp('press any key to continue'); pause;

% detect objects
boxes = detect(im, model, 0);


% get bounding boxes
bbox = getboxes(model, boxes);
top = nms(bbox, 0.5);
bbox = clipboxes(im, top);
% showboxes(im, bbox);
disp('bounding boxes');

