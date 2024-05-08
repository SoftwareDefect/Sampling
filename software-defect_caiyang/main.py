import experiment
import experiment_classifiers
import experiment_metrics

DATASETS = ["spring-integration", "broadleaf", "npm", "nova", "neutron", "brackets", "tomcat", "fabric", "jgroups","camel",'postgres','mozilla','platform','jdt','columba','bugzilla']


if __name__ == "__main__":

    # Comparative experiments of constructing Logistic Regression models on different metrics
    experiment_metrics.main(DATASETS)

    # using default LApredict classifier, timeperiod=2months, gap=2months
    timeperiod = 2
    model_para = 'default'
    experiment.main(DATASETS, timeperiod,model_para)

    # using default LApredict classifier, timeperiod=6months, gap=6months(Comparative experiments on time period)
    timeperiod = 6
    model_para = 'default'
    experiment.main(DATASETS,timeperiod,model_para )

    # using optimized LApredict classifier, timeperiod=2months, gap=2months(Comparative experiments on classifier parameter optimization)
    timeperiod = 2
    model_para = 'optimized'
    experiment.main(DATASETS,timeperiod,model_para )

    # using default NB classifier, timeperiod=2months(Comparative experiments on NB classifier)
    model = 'NB'
    experiment_classifiers.main(DATASETS,model)

    # using default RF classifier, timeperiod=2months(Comparative experiments on RF classifier )
    model = 'RF'
    experiment_classifiers.main(DATASETS, model)

