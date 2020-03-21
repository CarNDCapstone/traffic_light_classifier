# traffic_light_classifier

### Dataset

Copied over from [another Udacity repo](https://github.com/udacity/iSDC-P5-traffic-light-classifier-starter-code). As confirmed with Udacity mentors, this dataset is OK to use for the capstone project. The data could have been provided as a git submodule, but it was copied over since there was little data, and so that users don't have to deal with submodules, and in case the other repo gets deleted.

### Call to run 
```
python tf_classifier.py
```

### Sample Output

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #  
=================================================================
batch_normalization_1 (Batch (None, 128, 64, 3)        12        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 126, 62, 8)        224      
_________________________________________________________________
batch_normalization_2 (Batch (None, 126, 62, 8)        32        
_________________________________________________________________
activation_1 (Activation)    (None, 126, 62, 8)        0        
_________________________________________________________________
dropout_1 (Dropout)          (None, 126, 62, 8)        0        
_________________________________________________________________
average_pooling2d_1 (Average (None, 63, 31, 8)         0        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 61, 29, 16)        1168      
_________________________________________________________________
batch_normalization_3 (Batch (None, 61, 29, 16)        64        
_________________________________________________________________
activation_2 (Activation)    (None, 61, 29, 16)        0        
_________________________________________________________________
dropout_2 (Dropout)          (None, 61, 29, 16)        0        
_________________________________________________________________
average_pooling2d_2 (Average (None, 30, 14, 16)        0        
_________________________________________________________________
flatten_1 (Flatten)          (None, 6720)              0        
_________________________________________________________________
dense_1 (Dense)              (None, 120)               806520    
_________________________________________________________________
batch_normalization_4 (Batch (None, 120)               480      
_________________________________________________________________
activation_3 (Activation)    (None, 120)               0        
_________________________________________________________________
dropout_3 (Dropout)          (None, 120)               0        
_________________________________________________________________
dense_2 (Dense)              (None, 84)                10164    
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 255      
=================================================================
Total params: 818,919
Trainable params: 818,625
Non-trainable params: 294
_________________________________________________________________
None
Epoch 1/3
74/74 [==============================] - 3s - loss: 0.2232 - acc: 0.9181 - val_loss: 0.3395 - val_acc: 0.9192
Epoch 2/3
74/74 [==============================] - 2s - loss: 0.0963 - acc: 0.9738 - val_loss: 0.0887 - val_acc: 0.9630
Epoch 3/3
74/74 [==============================] - 1s - loss: 0.0764 - acc: 0.9730 - val_loss: 0.0427 - val_acc: 0.9899
Saved trained model at /home/workspace/traffic_light_classifier/tl_net
224/297 [=====================>........] - ETA: 0s('Test loss:', 0.042701885737604167)
('Test accuracy:', 0.98989898989898994)
Predicting one image took 3.249149 ms
```
