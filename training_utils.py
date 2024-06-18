import numpy as np

def generate_data(X, y, batch_size=32):
    num_samples = len(X)
    while True:
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch = X[batch_idx] / 255.0
            y_batch = y[batch_idx] / 255.0
            if y_batch.shape[-1] == 1:
                y_batch = np.repeat(y_batch, 3, axis=-1)
            yield X_batch, y_batch

def train_model(model, X_train, y_train, epochs=22):
    model.fit(generate_data(X_train, y_train), epochs=epochs, verbose=1, steps_per_epoch=len(X_train))
    return model
