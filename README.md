# Plane Classification

The objective is to predict the manufacturer and the family of an aircraft. On the one hand, I implement and compare a Convolutional Neural Network and a Support vector machine in the notebook `train_classification_model`. On the other hand, I implement a Transfer Learning apporach in the notebook `transfer_learning` by importing the weights of InceptionV3, that I compare with the initial Convolutional Neural Network. The dataset used contains more than 6000 aircraft images for about 30 classes of manufacturer.

Source of the dataset :

```
@techreport{maji13fine-grained,
   title         = {Fine-Grained Visual Classification of Aircraft},
   author        = {S. Maji and J. Kannala and E. Rahtu
                    and M. Blaschko and A. Vedaldi},
   year          = {2013},
   archivePrefix = {arXiv},
   eprint        = {1306.5151},
   primaryClass  = "cs-cv",
}
```

This study is developped into a Streamlit Dashboard translated in French at this link : https://share.streamlit.io/antoinepinto/planes-classification/main/app/app.py. The Dashboard is linked to the current repository with the application in the folder "app".

In the Dashboard, the user can choose to predict the manufacturer or the family and to choose which model to use (CNN, SVM or the model coming from Transfer Learning). 

![alt text](https://github.com/AntoinePinto/planes-classification/blob/main/images/screen1.png)

The probability distribution is displayed.

![alt text](https://github.com/AntoinePinto/planes-classification/blob/main/images/screen2.png)
