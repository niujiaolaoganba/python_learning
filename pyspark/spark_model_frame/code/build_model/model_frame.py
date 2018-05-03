#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pyspark import SparkContext,SparkConf
from pyspark import HiveContext
from pyspark.sql import functions as func
from config_read import ConfigInfo
from pyspark.ml import feature as fea
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml import Pipeline
from sklearn.metrics import precision_recall_curve,auc,classification_report
import pandas as pd
import numpy as np

class DataProcess(object):

    def __init__(self):
        conf=SparkConf().set('spark.sql.shuffle.partitions','50').set('spark.jars.packages','ml.combust.mleap:mleap-spark-base_2.11:0.7.0,ml.combust.mleap:mleap-spark_2.11:0.7.0')
        sc=SparkContext(conf=conf)
        sc.setLogLevel('WARN')
        self.hc=HiveContext(sc)

    def getRawData(self):
        sql=ConfigInfo('model_sql.sql').getSQL('pinyou_sql','app_geo_info')
        raw_data=self.hc.sql(sql)
        self.raw_data=raw_data.filter(func.col('user_pyid')!='null')
        self.raw_data.cache()

    def appNameFeature(self):
        app_name=self.raw_data.filter((func.col('app_id')!='null') | (func.col('app_name')!='null'))
        app_name=app_name.withColumn('appname_new',func.when(func.col('app_id')!='null',func.col('app_id')
                                                             ).when(func.col('app_name')!='null',func.col('app_name')
                                                                    ).otherwise(func.col('sign_platform')))

        top_app=app_name.groupBy('appname_new').agg(func.countDistinct('user_pyid'
                                                                       ).alias('user_cnt')
                                                    ).orderBy('user_cnt',ascending=0).limit(1000)
        top_app_list=top_app.select('appname_new').rdd.flatMap(lambda x:x).collect()
        app_name=app_name.filter(func.col('appname_new').isin(top_app_list))

        user_app=app_name.groupby('user_pyid').agg(func.concat_ws(',',func.collect_list(func.col('appname_new'))).alias('appnames'))
        user_app.write.format('orc').mode('overwrite').saveAsTable('default.cc_user_app')

    def getLabel(self):
        label_info=self.raw_data.filter(func.col('user_type')!='null')
        label_info=label_info.select('user_pyid','user_type').distinct()
        label_info.write.format('orc').mode('overwrite').saveAsTable('default.cc_label')

    def runFeatures(self):
        user_app=self.hc.sql('select * from default.cc_temp1')
        user_label=self.hc.sql('select * from default.cc_label')
        after_process=user_app.join(user_label,user_app['user_pyid']==user_label['user_pyid'],'inner').drop(user_label['user_pyid'])
        after_process.write.format('orc').mode('overwrite').saveAsTable('default.cc_after_process')

class CreateModel(object):

    def __init__(self):
        conf=SparkConf().set('spark.sql.shuffle.partitions','50').set('spark.jars.packages','ml.combust.mleap:mleap-spark-base_2.11:0.7.0,ml.combust.mleap:mleap-spark_2.11:0.7.0')
        sc=SparkContext(conf=conf)
        sc.setLogLevel('WARN')
        self.hc=HiveContext(sc)

    def getModelData(self):
        model_data=self.hc.sql('select * from default.cc_after_process')
        return model_data.filter(func.col('user_type').isin(['1','2']))

    def buildModel(self,save_pipe_path=None):
        df=self.getModelData()

        label_index=fea.StringIndexer(inputCol='user_type',outputCol='label')
        reTokenizer=fea.RegexTokenizer(inputCol='appnames',outputCol='appname_token',pattern=',')
        cnt_vector=fea.CountVectorizer(inputCol='appname_token',outputCol='appname_vector')
        vecAssembler = fea.VectorAssembler(inputCols=['appname_vector'], outputCol="feature")
        scaler=fea.StandardScaler(inputCol='feature',outputCol='features')

        if not save_pipe_path:
            lr=LogisticRegression()
            grid=ParamGridBuilder().addGrid(lr.elasticNetParam,[0,1]).build()
            evaluator=BinaryClassificationEvaluator(metricName="areaUnderPR")

            pipeline = Pipeline(stages=[label_index,reTokenizer, cnt_vector,vecAssembler,scaler])
            pipe = pipeline.fit(df)
            pipe_out=pipe.transform(df)

            cv=CrossValidator(estimator=lr,estimatorParamMaps=grid,evaluator=evaluator)
            model=cv.fit(pipe_out)

            print evaluator.evaluate(model.transform(pipe_out))
            print 'Best Param (regParam): ', model.bestModel._java_obj.getElasticNetParam()

            predict_result=model.transform(pipe_out).select('probability','label').toPandas()
            predict_result.to_csv('/home/chenchen/data/predict_result1.csv',index=False)
        else:
            lr=LogisticRegression(elasticNetParam=1.0)

            pipeline=Pipeline(stages=[label_index,reTokenizer, cnt_vector,vecAssembler,scaler,lr])
            model=pipeline.fit(df)

            model.save(save_pipe_path)
            print 'pipe saved'


    def modelEvalue(self,lookfor_threshold):
        df=pd.read_csv('/home/chenchen/data/predict_result.csv')
        df['probability']=df['probability'].apply(lambda x:x.strip('[]').split(','))
        predict_result=np.array(df['probability'].tolist(),dtype=np.float)

        precision,recall,pr_thresholds=precision_recall_curve(df['label'],
                                                              predict_result[:,1])
        thresholds=np.hstack(([0],pr_thresholds))
        print 'AUC:',auc(recall,precision)
        print 'precision:',precision
        print 'recall:',recall
        print 'threshold:',pr_thresholds

        print classification_report(df['label'],predict_result[:,1]>0.8)

        if lookfor_threshold:
            idx=precision>=lookfor_threshold
            print 'P=%.2f R=%.2f thresh=%.2f'%(precision[idx][0],recall[idx][0],thresholds[idx][0])