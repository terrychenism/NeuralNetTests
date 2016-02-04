import os
import tarfile

WORKING_DIR = './ILSVRC12'

os.listdir(WORKING_DIR + '/raw/')


os.mkdir(WORKING_DIR + '/ILSVRC2012_image_train_tar')
tarfile.TarFile(WORKING_DIR+'/raw/ILSVRC2012_img_train.tar').extractall(path=WORKING_DIR+'/ILSVRC2012_image_train_tar')
os.listdir(WORKING_DIR + '/ILSVRC2012_image_train_tar')


train_dir = WORKING_DIR + '/ILSVRC2012_img_train'
os.mkdir(train_dir)
for tar in [path for path in os.listdir(WORKING_DIR + '/ILSVRC2012_image_train_tar') if path.endswith('.tar')]:
    dirname = train_dir + '/' + tar.split('.')[0]
    os.mkdir(dirname)
    filename = WORKING_DIR + '/ILSVRC2012_image_train_tar/' + tar
    print 'extracting %s to %s' % (filename, dirname)
    f = tarfile.TarFile(filename)
    f.extractall(path=dirname)


