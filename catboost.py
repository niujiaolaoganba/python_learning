import catboost
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor


GradientBoostingRegressor()

class Score:
  def score_bagging(self, df):
      data = df.loc[((df.MODEL == 'getui') & (df.VERSION == '2018032802')) | ((df.MODEL == 'ipinyou') & (df.VERSION == '2017121218')),['DEVICE_ID','SCORE']]
      result = data.SCORE.groupby(data.DEVICE_ID).agg('mean')
      result['DEVICE_ID'] = result.index
      return result


from sklearn.preprocessing import normalize