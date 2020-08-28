# Team-6-ML-subsystem

##Tasks

1. Shape classification (torus vs not torus among torus,cylinder,cone,sphere,cuboid)
2. Colour classification (among black,brown,cyan,green,grey,navy blue,red,yellow)
3. Text Classification (among the above 8 colours)

1st,2nd and 3rd tasks are refered to as 1a,1b and 2 respectively.

##Datasets

All images were generated by randomly taking screenshots of solidwork models.

Link to solidwork models - 
https://drive.google.com/drive/folders/1KrcTThekn5h6yKdBS4yx_qBEnfn1EVU-?usp=sharing 

Link to all images - 
https://drive.google.com/drive/folders/1imAlxk4WJhky2T06ANjkfoIGZY7vl8zX?usp=sharing

Link to numpy arrays of images stored in .npy (easy to load and apply the model on)-
https://drive.google.com/drive/folders/1deNr7eN1qDWU-XvAQGWtSZn6meE_WHre?usp=sharing

##Things that you can find in this repo

1. Complete code which is present as **model{task}.ipynb**
1. Keras models trained from scratch which are abbreviated as **model{task}_{accuracy}.h5**.
1. Some images and videos to try this model.
1. Python files abbreviated as **ml_{task_name}.py**.
1. Tasks 1 and 2 can also be easily done using OpenCV and the file for the same is present as **cvshapes.py**.
1. You can also find a ROS script and if it runs fine on ROS python2 then please contact me (xD)!

##Usage 
1. If you want to take a look into the solid models or images, link to them is provided above.
2. If you want to evaluate the model on the whole dataset, I would suggest :
   1. Add the .npy files as referenced above (X_train.npy,y_train.npy,X_valid.npy,y_valid.npy)
	   to your drive.
   1. Open the corresponding colab notebook and import the necessary statements.
   1. Directly the load the numpy files from the section "load processed data".
   1. Then run the code `model.evaluate(X_valid,y_valid)`.
3. If you want to test the model on few images, I would suggest :
   1. download the corresponding python files and the model.
   1. Add the image path `cv2.imread()` and the model path in `model=tf.keras.models.load_model()` as an argument. 
   1. Run the python file from terminal using `python3 filename.py` .




