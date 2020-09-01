import multiprocessing
import tensorflow as tf
from tensorflow import keras
from data_prep import *
from unet_model import *
from datetime import datetime
import os

def run_model(learning_rate):
#     learning_rate = 0.001
    print("Training with learning rate: " + str(learning_rate))
    
    train_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "_lr" + str(learning_rate).split('.')[1]
    logdir = "logs/" + train_name
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    model = unet_model((512,512,1))

    opt = keras.optimizers.Adam(learning_rate=learning_rate) # default is 0.001
    model.compile(optimizer = opt, loss="binary_crossentropy", metrics=["accuracy"])

    # create_generators
    seed = 1
    batchsize = 2

    train_generator, val_generator = create_generators(seed=seed, batchsize=batchsize)
    # train_generator, val_generator = create_generators("tranformed_val", seed=seed, batchsize=batchsize)

    epochs = 50

    # history = model.fit(x=X_train, y=Y_train, epochs=30, batch_size=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=24/batchsize, # TODO: fix this #TODO: check that mult by 10 reduces number of epochs to get to same thing 
        validation_data=val_generator,
        validation_steps=6/batchsize, # TODO: fix this
        epochs=epochs,
        verbose=1,
        callbacks=[tensorboard_callback]
    )
    
    os.makedirs("saved_models/"+train_name+"/")
    model.save("saved_models/"+train_name+"/")

    print()
    
if __name__ == '__main__':
    learning_rates = [0.001, 0.0003, 0.0001]
#     learning_rates = [0.001]
    print("Number of learning rates to try: " + str(len(learning_rates)))
    for learning_rate in learning_rates:
        p = multiprocessing.Process(target=run_model, args=(learning_rate,))
        p.start()
        p.join()