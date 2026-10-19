from tensorflow import keras
from tensorflow.keras import layers

from common import *


def build_model(dims, dropout=0.2, branch_units=16, hidden_units=32):
    inputs, branches = [], []
    for name, dim in dims.items():
        inp = keras.Input(shape=(dim,), name=name)
        h = layers.Dense(branch_units, activation='relu')(inp)
        h = layers.Dropout(dropout)(h)
        inputs.append(inp)
        branches.append(h)
    merged = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    merged = layers.Dense(hidden_units, activation='relu')(merged)
    merged = layers.Dropout(dropout)(merged)
    output = layers.Dense(1, activation='sigmoid')(merged)
    model = keras.Model(inputs, output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def train_and_score(blocks, X_tr, y_tr, X_te, y_te, seed, dropout=0.2, batch=32, epochs=60, patience=6):
    keras.utils.set_random_seed(seed)
    scaler = StandardScaler().fit(X_tr)
    tr = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
    feed_tr = {name: tr[cols].to_numpy(np.float32) for name, cols in blocks.items()}
    feed_te = {name: te[cols].to_numpy(np.float32) for name, cols in blocks.items()}
    model = build_model({name: len(cols) for name, cols in blocks.items()}, dropout)
    model.fit(
        feed_tr, y_tr.to_numpy(), epochs=epochs, batch_size=batch, validation_split=0.15, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)],
    )
    p = model.predict(feed_te, verbose=0).ravel()
    return roc_auc_score(y_te, p), accuracy_score(y_te, (p >= 0.5).astype(int))


def make_blocks(prep, features, group_names=None):
    blocks = {}
    for name, cols in GROUPS.items():
        if group_names is not None and name not in group_names:
            continue
        selected = [c for c in cols if c in features]
        columns = prep.columns_of(selected)
        if columns:
            blocks[name] = columns
    return blocks
