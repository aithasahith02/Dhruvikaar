# Dhruvikaar
Student Verification System

✨The Dataset contains the images of the people properly dressed and improperly dressed for both training and testing.

✨Model is the file containing the code for Creating,training and testing the model.

✨Successful Execution of code will create the model file named as "new_model.h5".


The Docker Image that can deploy the model can be get using the command "docker pull sahithaitha02/dhruvikaar_centos:latest"
The docker image is based on the CentOS operating system and the dependencies for Centos:latest image to run the above model can be obtained by performing the follwing steps:

1->Launch the docker container from Centos image using the below commad.

👉docker run -it --nethost --env=DISPLAY --volume=$HOME/.Xauthority:/root/Xauthority --device = /dev/video0:/dev/video0 centos

This will provide the GUI and Camera access to the container from the docker host.



2->Then some more dependencies like python3 are needed and that can be obtained using yum or dnf commands.

👉yum install python3 -y

👉dnf install libglvnd-glx -y

👉 dnf install libcanberra-gtk* -y



3->The packages that are required to create and deploy the model are tensorflow,cv2,imutils and few more.These can be downloaded using pip and is done after upgrading the pip to the latest version.

The commands are..

👉python3 -m pip install --upgrade pip
   
   pip3 install scikit-learn 
   
   pip3 install scikit-build
   
   pip3 install opencv-python
   
   pip3 install tensorflow
   
   pip3 install matplotlib
   
   pip3 install imutils

All these dependencies can also be installed using a Dockerfile at a single go.
   

4->Once all the dependencies were satisfied the dataset can be downloaded from this repository along with the Code.


5->Finally the model is deployed using the command.

👉python3 VS.py
