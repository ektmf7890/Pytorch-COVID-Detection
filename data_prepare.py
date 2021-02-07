# Prepare train/valid(10)/test(30) set
import os
import shutil

def data_prepare():
    class_names = ['covid', 'normal', 'viral']
    root_dir = 'COVID-19RadiographyDatabase'
    source_dirs = ['COVID', 'NORMAL', 'Viral Pneumonia']

    # Creating directory struture
    if not os.path.exists(os.path.join(root_dir, 'test')):
        os.mkdir(os.path.join(root_dir, 'test'))

        # make folders for each class
        for class_name in class_names:
            if not os.path.exists(os.path.join(root_dir, 'test', class_name)):
                os.mkdir(os.path.join(root_dir, 'test', class_name))

    if not os.path.exists(os.path.join(root_dir, 'train')):
        os.mkdir(os.path.join(root_dir, 'train'))

        # make folders for each class
        for class_name in class_names:
            if not os.path.exists(os.path.join(root_dir, 'train', class_name)):
                os.mkdir(os.path.join(root_dir, 'train', class_name))

    if not os.path.exists(os.path.join(root_dir, 'valid')):
        os.mkdir(os.path.join(root_dir, 'valid'))

        # make folders for each class
        for class_name in class_names:
            if not os.path.exists(os.path.join(root_dir, 'valid', class_name)):
                os.mkdir(os.path.join(root_dir, 'valid', class_name))


    #  Rename source directories
    for i, d in enumerate(source_dirs):
        if os.path.exists(os.path.join(root_dir, source_dirs[i])):
            os.rename(os.path.join(root_dir, source_dirs[i]), os.path.join(root_dir, class_names[i]))


    # Sample valid/test set images
    import random

    num_valid_images = 10
    num_test_images = 30

    for class_name in class_names:
        images = [x for x in os.listdir(os.path.join(root_dir, class_name)) if x.lower().endswith('png')]

        # randomly select 10 images for valid set and move
        valid_images = images[:num_valid_images]
        images = images[num_valid_images:]
        for img in valid_images:
            shutil.move(os.path.join(root_dir, class_name, img), os.path.join(root_dir, 'valid', class_name))

        # randomly select 30 images for test set
        test_images = images[:num_test_images]
        for img in test_images:
            shutil.move(os.path.join(root_dir, class_name, img), os.path.join(root_dir, 'test', class_name)) 

    
    # Move remaining images to train set
    for class_ in class_names:
        shutil.move(os.path.join(root_dir, class_), os.path.join(root_dir, 'train', class_))
