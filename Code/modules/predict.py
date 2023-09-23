from typing import List, Optional, Union, Literal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import base

def predict_original_files(path: Path, 
                           filetype: str, 
                           scale: bool = True,
                           model: Union[str, Path] = None,
                           true_labels: Optional[List[str]] = None,
                           N_points: int = 1024,
                           flip: bool = True,
                           energy: Literal['kinetic', 'binding'] = 'kinetic',
                           ) -> (np.array, np.array, np.array, np.array, List[str], plt.figure):
    '''
    input:
        path: path to file
        filetype: 'vms' or 'csv'
        model: path to model
        scale: scale data to [0,1]
        true_labels: list of true labels, e.g. ['Ag', 'Al']

    output:
        x: x values of original file
        y: y values of original file
        x_new: x values of transformed file
        y_new: y values of transformed file
        pr: predicted labels
        fig: figure with plot
    '''
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from preprocess import parse_file
    from sklearn.preprocessing import MultiLabelBinarizer
    import base
    
    elements = base.load_elem()
    n_elements = len(elements)

    mlb = MultiLabelBinarizer()
    mlb.fit([elements])
    labs = pd.DataFrame(mlb.fit_transform([elements]), columns=mlb.classes_)

    x, y, x_new, y_new = parse_file(path, filetype, scale=scale, N_points=N_points, flip=flip, energy=energy)

    fig, ax = plt.subplots(1,3, figsize=(30,5), gridspec_kw={'width_ratios': [1,4,1]})
    fig.suptitle(f'Prediction of file {path}', fontsize=16)
    ax[0].set_title('Original')
    ax[0].plot(x, y)
    ax[0].set_xlim(min(x), max(x))

    ax[2].set_title('Interpolated / Transformed')
    ax[2].plot(x_new, y_new)
    ax[2].set_xlim(min(x_new), max(x_new))

    model = tf.keras.models.load_model(model)
    p = model.predict(y_new.reshape(1,1,N_points), batch_size=1)

    top = np.zeros(n_elements)
    bot = np.zeros(n_elements)

    top[p[0].argmax()] = int(1)
    bot[p[1].argmax()] = int(1)

    top = top.astype(int) 
    bot = bot.astype(int)

    pred = np.array([top, bot])
    pr = mlb.inverse_transform(pred)
    print(f'Prediction: \t  \t {pr[1][0]} on {pr[0][0]}')

    ax[1].plot(p[0][0][0]);
    ax[1].plot(p[1][0][0], '--');
    ax[1].legend(['predicted_bot', 'predicted_top', 'true_bot', 'true_top']);
    ax[1].set_xticks([i for i in range(len(elements))],
                     labels=labs.columns.values);
    ax[1].xaxis.set_tick_params(labelsize=9)

    if true_labels:
        tr = mlb.transform([(true_labels[0],), (true_labels[1],)])
        ax[1].plot(tr[0], 'g+', alpha=0.5)
        ax[1].plot(tr[1], 'rx', alpha=0.5)
        ax[1].legend(['predicted_bot', 'predicted_top', 'true_bot', 'true_top']);

    return x, y, x_new, y_new, pr, fig


def predict_from_array(x: np.array,
                       labels: np.array,
                       shape, model) -> List[tuple]:
    # shape = (x.shape[0], 1, 1024)
    mlb, _ = base.retreive_mlb_and_elements()
    pr = model.predict(x.reshape(shape[0],
                                 shape[1],
                                 shape[2]),
                                 batch_size=shape[0])
    
    predicted_label = list(range(len(x)))
    for i, predictions in enumerate(pr):
        # print(i, predictions.shape, predictions.argmax())
        top_prediction = predictions.argmax()
        correct = (top_prediction == mlb.transform([[labels[i]]]).argmax())  # prediction correct?
        # print(top_prediction, mlb.transform([[labels[i]]]).argmax(), correct)
        
        # prediction label
        b = np.zeros_like(predictions)
        # print(b.shape, b.ndim)
        b[top_prediction] = 1
        b = b.astype(int)
        predicted_label[i] = (correct, mlb.inverse_transform(np.array([b])), labels[i])
        
    return predicted_label

def predict_from_array_h5(x: np.array, labels, shape, model):
    import tensorflow as tf
    mlb, _ = base.retreive_mlb_and_elements()
    x = x.reshape(shape[0], shape[1], shape[2])
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    pr = model.predict(x, batch_size=shape[0],verbose=None)
    predicted_label = list(range(len(labels)))
    for i, predictions in enumerate(pr):
        top_prediction = predictions[0].argmax()
        correct = (top_prediction == mlb.transform([[labels[i]]]).argmax())  # prediction correct?
        # print(top_prediction, mlb.transform([[labels[i]]]).argmax(), correct)
        
        # prediction label
        a = predictions[0]
        b = np.zeros_like(a)
        b[a.argmax()] = 1
        b = b.astype(int)
        predicted_label[i] = (correct, mlb.inverse_transform(np.array([b])))
    return predicted_label