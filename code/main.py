from interface import *
from model import *


if __name__ == "__main__":
    model = GHGPredictor()
    model.load_bics_list("/Users/ysc/Desktop/Capstone/BICS list.csv")
    model.load_data("/Users/ysc/Desktop/Capstone/data.csv")
    model.train_models(nrestarts=5)
    model.train_models_2(fit_intercept=True)
    main(model)