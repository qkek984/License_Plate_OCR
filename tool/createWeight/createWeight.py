from keras.models import load_model
model = load_model('stringVModel.h5')
model.save_weights('stringVModelWeight.h5')
model.summary()