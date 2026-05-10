from sklearn.ensemble import RandomForestClassifier


def train_random_forest(x_train, y_train, random_state=42):
    """Train a small Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)
    return model


