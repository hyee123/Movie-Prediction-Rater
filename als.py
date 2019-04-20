import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import tree
from scipy.sparse import dok_matrix

from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import AlgoBase
from surprise import BaselineOnly
from surprise import SVD
from surprise import Prediction


# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getTrainDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    y = dataframe['rating']
    x = dataframe.drop('rating', axis=1)
    #uid = dataframe['userId'].to_frame()
    #iid = dataframe['movieId'].to_frame()
    uid = str(6)
    iid = str(300)
    return x, y, uid, iid


# Reads the data from CSV files, converts it into Dataframe and returns x and y dataframes
def getTestDataframe(filePath):
    dataframe = pd.read_csv(filePath)
    ids = dataframe['Id']
    user = dataframe['userId']
    movies = dataframe['movieId']


    return ids, user, movies


def getMatrix(x, y):
    user_ids = x['userId']
    movie_ids = x['movieId']
    num_users = np.max(user_ids)
    num_movies = int(np.max(movie_ids))


    print(num_users)
    print(num_movies)


    print("TYPE:", type(user_ids))


    q = dok_matrix((num_users, num_movies))


    for i in range(x.shape[0]):
        q[user_ids[i], movie_ids[i]] = y[i]
        if i % 10000 == 0:
            print("Iteration: ", i, i / 11000000.0, "% of the way there")


    print("q", q)
    return q




# predicted_y and test_y are the predicted and actual y values respectively as numpy arrays
# function prints the mean squared error value for the test dataset
def compute_rmse(predicted_y, y):
    rmse = np.sum((predicted_y - y)**2)/predicted_y.shape[0]
    return np.sqrt(rmse)


# Linear Regression implementation
class AlternatingLeastSquares(object):
    # Initializes by reading data, setting hyper-parameters, and forming linear model
    # Forms a linear model (learns the parameter) according to type of beta (0 - closed form, 1 - batch gradient, 2 - stochastic gradient)
    # Performs z-score normalization if z_score is 1
    def __init__(self):
        self.train_x = pd.DataFrame()
        self.train_y = pd.DataFrame()
        self.val_x = pd.DataFrame()
        self.val_y = pd.DataFrame()
        self.predicted_y = list()
        self.y = np.zeros(100)
        self.reg = LinearRegression()
        self.knn = AlgoBase()
        #self.train_data =
        #self.val_data =
        #self.test_data =


    def load_data(self, train_file, val_file, test_file):
        #self.train_x, self.train_y = getTrainDataframe(train_file)
        #self.val_x, self.val_y, self.val_uid, self.val_iid = getTrainDataframe(val_file)
        self.test_ids, self.test_users, self.test_movies = getTestDataframe(test_file)

        train_reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
        train_data = Dataset.load_from_file(train_file, train_reader)
        #self.trainset1 = train_data.build_full_trainset()
        self.trainset, gobruins = train_test_split(train_data, test_size=0.05)
        print(type(self.trainset))
        #val_data = Dataset.load_from_file(val_file, train_reader)
        #gobruins, self.valset = train_test_split(val_data, test_size=1.0)
        #print(self.valset[0])

        d = {'1user': [str(item) for item in self.test_users], '2item': [str(item) for item in self.test_movies], '3rating': self.test_ids}
        df = pd.DataFrame(data=d)
        print(df)
        test_data = Dataset.load_from_df(df, train_reader)

        test_trainset = test_data.build_full_trainset()
        self.testset = test_trainset.build_testset()

        #obruins, self.testset = train_test_split(test_data, test_size=1.0)
        print(self.testset[0])


        #test_data = Dataset.load_from_file(test_file, train_reader)

        #self.q = getMatrix(self.train_x, self.train_y)
        #self.rp = self.train_df.pivot_table(columns=['movieId'],index=['userId'],values='rating')
        #self.rp = self.train_df.groupby(['userId', 'movieId'])['rating'].unstack()
        #print("HEAD:", self.rp.head())


    # Gets the beta according to input
    def train(self):


        #self.reg = LinearRegression().fit(self.train_x, self.train_y)
        self.knn = SVD()
        self.knn.fit(self.trainset)
        #self.reg = tree.DecisionTreeRegressor().fit(self.train_x, self.train_y)


        #train_mse = compute_rmse(predicted_y, self.train_y.values)
        print('Training MSE: ', train_mse)


        return 0


    # Predicts the y values of all test points
    # Outputs the predicted y values to the text file named "logistic-regression-output_algType_isNormalized" inside "output" folder
    # Computes MSE
    def val_rmse(self):
        #prediction = self.reg.predict(self.val_x)
        prediction = self.knn.test(self.valset, verbose=True)
        #print(prediction)
        rmse = accuracy.rmse(prediction)
        #rmse = compute_rmse(self.prediction, self.val_y.values)
        return rmse


    def predict(self):
        self.predicted_y = self.knn.test(self.testset, verbose=False)
        print(self.predicted_y[3][3], self.predicted_y[4][3])
        self.log('als')
        print("Logged")
        #uid=user iid=movie rui=id, est=values output
        #sort by rui, then output est
        #self.predicted_y = self.reg.predict(self.test_x)
        #self.predicted_y = self.knn.test(self.test_whole)

    def log(self, file_name):
        #output = np.zeros((self.test_ids.shape[0], 2))
        #output[:, 0] = self.test_ids
        #output[:, 1] = self.predicted_y


        print("Logging")
        predicted_y_est = [item[3] for item in self.predicted_y]

        d = {'Id': self.test_ids.values, 'rating': predicted_y_est}

        df = pd.DataFrame(data=d)
        df.to_csv(file_name + "_output.csv", index=None)

        # f = open("linear-regression-output.csv", "w")
        #f = open("knn-output.csv", "w")
        #f.write(output)






if __name__ == '__main__':
    # Change 1st paramter to 0 for closed form, 1 for batch gradient, 2 for stochastic gradient
    # Add a second paramter with value 1 for z score normalization
    # Add a second parameter with value 1 for regularization


    print("I'm running")
    als = AlternatingLeastSquares()
    print("Object instantiated")
    als.load_data('train_ratings.csv', 'val_ratings.csv', 'test_ratings.csv')


    print("Data Loaded")
    # training
    beta = als.train()

    als.predict()

    #rmse = als.val_rmse()




    #print("Validation RMSE:", rmse)
    #print("Our Predictions:", als.predicted_y)




    # testing
    test_rmse = als.predict(beta)
    print('Test RMSE: ', test_rmse)