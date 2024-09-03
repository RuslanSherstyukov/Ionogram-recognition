
"""

@author: Dr. Ruslan Sherstyukov, Sodankyla Geophysical Observatory, 2024

"""


import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def set_path():
    
    models_directory = os.path.join(os.getcwd(), "Models")
    ionograms_directory = os.path.join(os.getcwd(), "2021P")
    ground_truth_directory = os.path.join(os.getcwd(), "GroundTruth")
    return models_directory,ionograms_directory,ground_truth_directory

def param_to_pixel(param,h=True):
    if h:
        pixel = np.around(0.4876*(param - 0)/2.8617, decimals=0)
    else:
        pixel = np.around(0.4339*(param - 0.5262)/0.0262, decimals=0)
    return int(pixel)

def load_models():
    
    models_directory,ionograms_directory,ground_truth_directory = set_path()

    ModelCatF=tf.keras.models.load_model(os.path.join(models_directory, "F_class.h5"))
    ModelCatE=tf.keras.models.load_model(os.path.join(models_directory, "E_class_new.h5"))
    ModelFmin=tf.keras.models.load_model(os.path.join(models_directory, "FMIN_all_PP.h5"))
    ModelFoF2=tf.keras.models.load_model(os.path.join(models_directory, "FOF2_all_PP.h5"))
    ModelFoF1FoF2=tf.keras.models.load_model(os.path.join(models_directory, "FOF1_FOF2_all_PP.h5"))
    ModelFoF1=tf.keras.models.load_model(os.path.join(models_directory, "FOF1_all_PP.h5"))
    ModelFoE=tf.keras.models.load_model(os.path.join(models_directory, "FOE_all_PP.h5"))
    ModelFoEs=tf.keras.models.load_model(os.path.join(models_directory, "FOES_all_PP_new.h5"))
    ModelFoEs_K=tf.keras.models.load_model(os.path.join(models_directory, "FOES_K_all_PP.h5"))
    ModelFbEs=tf.keras.models.load_model(os.path.join(models_directory, "FBES_all_PP_new.h5"))
    ModelHF=tf.keras.models.load_model(os.path.join(models_directory, "HF_all_PP.h5"))
    ModelHE=tf.keras.models.load_model(os.path.join(models_directory, "HE_all_PP.h5"))
    ModelHEs=tf.keras.models.load_model(os.path.join(models_directory, "HES_all_PP.h5"))

    Models = {"CATF":ModelCatF, "CATE":ModelCatE, 
              "FMIN":ModelFmin,"HES":ModelHEs, "FOES":ModelFoEs, "FOESK":ModelFoEs_K, "FBES":ModelFbEs,
              "HE":ModelHE, "FOE":ModelFoE,
              "HF":ModelHF, "FOF1":ModelFoF1,
              "FOF1FOF2":ModelFoF1FoF2, "FOF2":ModelFoF2}
    return Models


def Evaluate(parameter = "FOF2", model = "FOF2"):
    
    models_directory,ionograms_directory,ground_truth_directory = set_path()
    Models = load_models()

    data = pd.read_csv(os.path.join(ground_truth_directory, f"{parameter}.csv"))
    file_name = data['FILE_NAME']
    data['FILE_NAME'] = data['FILE_NAME'].apply(lambda x: os.path.join(ionograms_directory, x))
    image_names = data.iloc[:, 0] 
    values = data.iloc[:, 1] 

    # Load images as arrays
    images = []
    for image_name in image_names:#[0:-1]:
        img = load_img(image_name, target_size=(256, 256),color_mode='grayscale')
        img = img_to_array(img)
        images.append(img)
        
    # Convert to numpy array 
    X = (np.array(images).astype('float32')) / 255.
    y = np.array(values).astype('float32')
    y = np.expand_dims(y, axis=-1)

    # Calculate metrics
    predictions = Models[model].predict(X)
    diff = y - predictions
    MAE = np.mean(np.abs(diff[:,0]))
    RMSE = np.sqrt(np.mean(np.square(diff)))

    # Convert to pandas dataframe
    MAE = pd.DataFrame([MAE] + [None] * (len(X)-1), columns=['MAE'])
    RMSE = pd.DataFrame([RMSE] + [None] * (len(X)-1), columns=['RMSE'])
    predictions = pd.DataFrame(predictions, columns=['Predictions'])
    y = pd.DataFrame(y, columns=['GroundTruth'])
    diff = pd.DataFrame(diff, columns=['Difference'])

    # Save dataframe
    df = pd.concat([file_name,predictions,y,diff,MAE,RMSE], axis=1)                 
    df.to_csv(f"Evaluation_{parameter}.csv", index=True, header=True)

     
def IonogramShow(parameter = "FOF2", ionogram_index = 50):

    
    data = pd.read_csv(os.path.join(os.getcwd(), f"Evaluation_{parameter}.csv"))
    models_directory,ionograms_directory,ground_truth_directory = set_path()
    ionogram_name = (os.path.join(ionograms_directory, f"{data['FILE_NAME'][ionogram_index]}"))
    
    if parameter in ["HE", "HES", "HF"]:
        Prediction=param_to_pixel(data["Predictions"][ionogram_index],h=True)
        GroundTruth=param_to_pixel(data["GroundTruth"][ionogram_index],h=True)
    else:
        Prediction=param_to_pixel(data["Predictions"][ionogram_index],h=False)
        GroundTruth=param_to_pixel(data["GroundTruth"][ionogram_index],h=False)

     # Load image 
    ionogram = load_img(ionogram_name, target_size=(256, 256),color_mode='grayscale')
    ionogram = img_to_array(ionogram)

    # Display the image
    fig, axs1 = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    axs1.pcolor(ionogram[:,:,0])
    
    if parameter in ["HE", "HES", "HF"]:
        plt.plot(np.arange(0,256,4),64*[Prediction],'.w',markersize=15)
        plt.plot(np.arange(2,256,4),64*[GroundTruth],'.r',markersize=15)
        axs1.set_xlabel('frequency (MHz)', fontsize=30)
        axs1.set_ylabel('virtual height (km)', fontsize=30)
    else:
        plt.plot(64*[Prediction],np.arange(0,256,4),'.w',markersize=15)
        plt.plot(64*[GroundTruth],np.arange(2,256,4),'.r',markersize=15)
        axs1.set_xlabel('frequency (MHz)', fontsize=30)
        axs1.set_ylabel('virtual height (km)', fontsize=30)
        
    axs1.set_xticks(ticks=np.arange(0, 256, 1 / 0.0262 * 0.4339))
    axs1.set_xticklabels(np.arange(0.5, 256 * 0.0262 / 0.4339 + 0.5262, 1))
    axs1.set_yticks(ticks=np.arange(0, 256, 100 / 2.8617 * 0.4876))
    axs1.set_yticklabels(np.arange(0, 256 * 2.8617 / 0.4876, 100))
    axs1.tick_params(axis='x', labelsize=30)
    axs1.tick_params(axis='y', labelsize=30)
    axs1.set_title(data['FILE_NAME'][ionogram_index], fontsize=40)
    plt.show()
    fig.savefig(f"{data['FILE_NAME'][ionogram_index]}")


    

if __name__ == "__main__":
    print("ModelsEvaluation.py is running directly.")

