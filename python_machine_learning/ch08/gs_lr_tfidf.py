"""
Logistic regression, 5-fold stratified cross-validation, from book example.
"""


def main():
    df = pd.read_csv('./movie_data.csv')
