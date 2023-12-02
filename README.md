![Fabric](https://i.pinimg.com/originals/e3/fa/32/e3fa32b8d27bbb6f39161378dbe9e0c9.gif)

# _**Identifying Defects-in-the-Various-Fabrics-using-Convolutional-Neural-Networks**_
The main objective of this project is to identify whether there is any fault in the fabrics or not, this is done using convolutional neural networks where the model is trained on both defected and non-defect fabric dataset. The dataset was obtained from private data repository and some pre-processing has been performed to make sure the data is clean for modelling.

# _**Base Paper**_
+ https://www.researchgate.net/publication/316687014_Fabric_Fault_Detection_Using_Image_Processing

# _**Algorithm Description**_
So, we have used Convolutional neural networks to identify whether a person has brain tumour or not, as we all know, how sophisticated CNNs are and how they can learn almost anything like a brain does, this can help us save a lot of time and also giving almost accurate predictions for the disease. As we discussed convolutional neural networks are very sophisticated and more advanced version of neural networks, these are very superior to other neural networks which works better with images and audio/speech input signal. A CNN network comprises of 3 important layers such as a convolutional layer, pooling layer and fully connected layer. we can have as many layers as possible depending on the domain and project we are working on.

# _**Reference:**_

+ https://www.ibm.com/cloud/learn/convolutional-neural-networks

![Neural Network](https://editor.analyticsvidhya.com/uploads/90650dnn2.jpeg)

# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://1.bp.blogspot.com/-UJ1Ws2zZ9V4/TtMbG2ynJiI/AAAAAAAABbM/m6t2kuEhKdY/s1600/The-biggest-anaconda-snake-3.jpg)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i0.wp.com/reptileworldfacts.com/wp-content/uploads/2019/05/male-blonde-super-tiger-reticulated-python.jpg?resize=351%2C351&ssl=1)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

# _**Let us now see how to create an environment in Anaconda.**_
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd C:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

![Happy](https://media0.giphy.com/media/l1J9vjZgVNYsSTTeo/giphy.gif)  

# _**Steps to execute**_
**Note:** Make sure you have added path while installing the software’s.

1.	Install the prerequisites/software’s required to execute the code.
2.	Press windows key and type in anaconda prompt a terminal opens up.
3.	Before executing the code, we need to create a specific environment which allows us to install the required libraries necessary for our project.
•	Type conda create -name “env_name”, e.g.: conda create -name project_1
•	Type conda activate “env_name, e.g.: conda activate project_1
4.	Make sure you are in the correct path in your terminal, where you have saved your executable file/folder. E.g.: cd A:\project\AI\Completed\project_name, then press enter.
5.	Install necessary libraries from requirements.txt file provided.
6.	Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
7.  Run main.ipynb run the final code, and make sure to change the path of the model and image folders. You can even see preprocess.ipynb to get a feel of how the images are pre-processed.

# _**Data Description**_
So, the dataset in the project was collected from a private repository and some were collected from Kaggle data repository. It consists of two classes of images each class consists of more than 100 images. Defected fabric and normal. Below are some sample images of 2 classes. Create your own Dataset from sources like kaggle, google etc.

**Normal**

![Normal](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.zNLqVGx1oDkoPaPm8iku8gAAAA%26pid%3DApi&f=1)

**Defect**

![Defect](https://i0.wp.com/garmentsmerchandising.com/wp-content/uploads/2016/03/Broken-pick.png)

# _Issues Faced._
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python or 3.8, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.
4. Make sure to change the paths of the model and dataset in the code.

# _Note:_
**All the required data hasn't been provided over here. Please feel free to contact me for dataset or any issues.**

### **Let’s Connect**

<a href="https://linkedin.com/in/mudassiruddin21" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="mudassiruddin21" height="30" width="40" /></a>

![Connect](https://media0.giphy.com/media/l9ZWFT0IjZbzi/200w.gif)

# ___**Yes, you now have more knowledge than yesterday, Keep Going.**___
![Congrats](https://media1.tenor.com/images/43d14c43c382da190601052f59a87072/tenor.gif?itemid=4115466)
