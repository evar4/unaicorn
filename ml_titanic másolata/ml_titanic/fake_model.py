def fake_predict(user_age):
    if user_age > 10:
        prediction = "survived (over 10)"
    else:
        prediction = "super survived (under 10)"
    return prediction
