import pandas as pd
from sklearn.metrics import log_loss
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('./data/current/numerai_training_data.csv')

train, test = cross_validation.train_test_split(data,
                                                test_size=0.9,
                                                random_state=0)

features = ["feature1", "feature2", "feature3", "feature4", "feature5",
            "feature6", "feature7", "feature8", "feature9", "feature10",
            "feature11", "feature12", "feature13", "feature14", "feature15",
            "feature16", "feature17", "feature18", "feature19", "feature20",
            "feature21"]

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=5)
model.fit(train[features], train['target'])
test_predictions = model.predict_proba(test[features])
print("Test Log Loss %f" % (log_loss(test['target'], test_predictions)))


tournament_data = pd.read_csv('./data/current/numerai_tournament_data.csv')
tournament_predictions = model.predict_proba(tournament_data[features])

result = tournament_data
result['probability'] = tournament_predictions[:,1]

result.to_csv("./output/out-rf.csv",
              columns= ('t_id', 'probability'),
              index=None)
